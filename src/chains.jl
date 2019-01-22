module Chains

using ArgCheck: @argcheck
using Parameters: @unpack
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, TYPEDEF

####
#### Index layouts
####

"""
$(TYPEDEF)
"""
struct IndexLayout{D <: Tuple{Vararg{Int}}}
    offset::Int
    len::Int
    dims::D
    function IndexLayout(offset, len, dims::D) where D
        @argcheck offset ≥ 0
        @argcheck len > 0
        @argcheck prod(dims) == len
        new{D}(offset, len, dims)
    end
end

function Base.getindex(A::AbstractVector, layout::IndexLayout)
    @unpack offset, len, dims = layout
    if isempty(dims)
        A[offset + 1]
    else
        reshape(view(A, offset .+ (1:len)), dims)
    end
end

function _getindex_stage(A, row_index, col_index)
    ix, _ = to_indices(A, (row_index, Colon()))
    if ix isa Integer
        A[ix, :][col_index]
    else
        map(row -> row[col_index], eachrow(view(A, ix, :)))
    end
end

function Base.getindex(A::AbstractMatrix, row_index, layout::IndexLayout)
    _getindex_stage(A, row_index, layout)
end

# optimized case for scalar layouts
function Base.getindex(A::AbstractMatrix, row_index, layout::IndexLayout{Tuple{}})
    col_index = layout.offset + 1
    ix, _ = to_indices(A, (row_index, Colon()))
    if ix isa Integer
        A[ix, col_index]
    else
        view(A, ix, col_index)
    end
end

_dims_string(dims::Tuple{}) = "scalar"
_dims_string(dims::Tuple{Int}) = "Vector($(first(dims)))"
_dims_string(dims) = "Array$(dims)"

function Base.show(io::IO, layout::IndexLayout)
    @unpack offset, len, dims = layout
    printstyled(io, "$((offset+1):(offset+len)) "; color = :light_black)
    printstyled(io, _dims_string(dims); color = :cyan)
end

"""
Type alias for a NamedTuple of array dimensions. Not exported.
"""
const NamedDims = NamedTuple{N,T} where {N,T<:Tuple{Vararg{Tuple{Vararg{Int}}}}}

"""
$(SIGNATURES)

Calculate column layouts from a `NamedTuple`) of array sizes.
"""
function _column_layouts(named_dims::NamedDims)
    acc = 0
    layouts = map(named_dims) do dims
        len = prod(dims)
        layout = IndexLayout(acc, len, dims)
        acc += len
        layout
    end
    layouts
end

####
#### Column schema
####

struct ColumnSchema{T}
    layouts::T
    function ColumnSchema(named_dims::NamedDims)
        layouts = _column_layouts(named_dims)
        new{typeof(layouts)}(layouts)
    end
end

function Base.show(io::IO, cs::ColumnSchema)
    get(io, :typeinfo, true) && print(io, "Column schema with layouts")
    for (name, layout) in pairs(layouts(cs))
        printstyled(io, "\n    ", name; color = :blue)
        print(io, " ", layout)
    end
end

layouts(cs::ColumnSchema) = getfield(cs, :layouts)

Base.getproperty(cs::ColumnSchema, name::Symbol) = getfield(layouts(cs), name)

Base.propertynames(cs::ColumnSchema) = propertynames(layouts(scs))

function Base.length(cs::ColumnSchema)
    last_layout = layouts(cs)[end]
    last_layout.offset + last_layout.len
end

function Base.getindex(V::AbstractVector, cs::ColumnSchema)
    @argcheck !Base.has_offset_axes(V)
    @argcheck length(cs) == length(V)
    map(layout -> V[layout], layouts(cs))
end

function Base.getindex(A::AbstractMatrix, row_index, cs::ColumnSchema)
    _getindex_stage(A, row_index, cs)
end

####
#### Chain
####

struct Chain{S <: ColumnSchema, M <: AbstractMatrix}
    schema::S
    sample_matrix::M
    thinning::Int
    warmup::Int
    is_ordered::Bool
    function Chain(schema::S, sample_matrix::M;
                   thinning::Int = 1, warmup::Int = 0,
                   is_ordered::Bool = false) where {S <: ColumnSchema, M <: AbstractMatrix}
        @argcheck !Base.has_offset_axes(sample_matrix)
        @argcheck length(schema) == size(sample_matrix, 2)
        @argcheck thinning ≥ 1
        @argcheck warmup ≥ 0
        new{S,M}(schema, sample_matrix, thinning, warmup, is_ordered)
    end
end

"""
$(SIGNATURES)

A `NamedTuple` of `IndexLayout`s for views into the sample matrix.
"""
schema(chain::Chain) = chain.schema

function Base.show(io::IO, chain::Chain)
    @unpack schema, sample_matrix, thinning, warmup, is_ordered = chain
    print(io, is_ordered ? "Ordered" : "Unordered", " MCMC chain of ")
    printstyled(io, "$(size(sample_matrix, 1)) rows"; color = :red)
    is_ordered && print(io, " of which $(warmup) are warmup, thinning $(thinning)")
    print(IOContext(io, :typeinfo => false), " with schema", schema)
end

function sample_matrix(chain::Chain, include_warmup::Val{false})
    @view chain.sample_matrix[(chain.warmup + 1):end, :]
end

sample_matrix(chain::Chain, include_warmup::Val{true}) = chain.sample_matrix

"""
$(FUNCTIONNAME)(chain, include_warmup = Val{false}())

The sample matrix, with or without the warmup.
"""
sample_matrix(chain::Chain) = sample_matrix(chain, Val{false}())

"""
$(SIGNATURES)

The thinning for the sample. When `is_ordered(chain) == false`, it returns `nothing`.
"""
function thinning(chain::Chain)
    @unpack is_ordered, thinning = chain
    is_ordered ? thinning : nothing
end

"""
$(SIGNATURES)

Return the number of samples from the beginning which should be considered warmup.
"""
warmup(chain::Chain) = chain.warmup

"""
$(SIGNATURES)

Return `true` iff the chain is ordered, ie its columns can be considered consecutive draws
from a Markov chain.
"""
is_ordered(chain::Chain) = chain.is_ordered

function Base.vcat(chain1::Chain, chains::Chain...)
    schema1 = schema(chain1)
    @argcheck all(c -> schema(c) == schema1, chains)
    Chain(schema1, mapreduce(sample_matrix, vcat, (chain1, chains...));
          thinning = 1, is_ordered = false, warmup = 0)
end

function Base.getindex(chain::Chain, row_index, name::Symbol)
    getindex(sample_matrix(chain), row_index, getproperty(schema(chain), name))
end

function Base.getindex(chain::Chain, row_index, ::Colon)
    getindex(sample_matrix(chain), row_index, schema(chain))
end

Base.eachcol(chain::Chain) = eachcol(sample_matrix(chain))

end
