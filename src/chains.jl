module Chains

using ArgCheck: @argcheck
using Parameters: @unpack
using DocStringExtensions: FUNCTIONNAME, SIGNATURES

struct ColumnLayout{D <: Tuple{Vararg{Int}}}
    offset::Int
    len::Int
    dims::D
    function ColumnLayout(offset, len, dims::D) where D
        @argcheck offset ≥ 0
        @argcheck len > 0
        @argcheck prod(dims) == len
        new{D}(offset, len, dims)
    end
end

function Base.view(A::AbstractMatrix, layout::ColumnLayout)
    @unpack offset, len, dims = layout
    reshape(view(A, Colon(), offset .+ (1:len)), (size(A, 1), dims...))
end

function Base.view(A::AbstractVector, layout::ColumnLayout)
    @unpack offset, len, dims = layout
    if isempty(dims)
        A[offset + 1]
    else
        reshape(view(A, offset .+ (1:len)), dims)
    end
end

"""
$(SIGNATURES)

Calculate a schema from a collection (preferably a `NamedTuple`) of array sizes.
"""
function calculate_schema(multiple_dims)
    acc = 0
    map(multiple_dims) do dims
        len = prod(dims)
        layout = ColumnLayout(acc, len, dims)
        acc += len
        layout
    end
end

"""
$(SIGNATURES)

Test if `schema` is valid, and has the given total length.
"""
function is_valid_schema(schema, total_len)
    acc = 0
    for layout in schema
        layout.offset == acc || return false
        acc += layout.len
    end
    acc == total_len
end

struct Chain{L <: NamedTuple, M <: Matrix}
    schema::L
    sample_matrix::M
    thinning::Int
    warmup::Int
    is_ordered::Bool
    function Chain(schema::L, sample_matrix::M;
                   thinning::Int = 1, warmup::Int = 0,
                   is_ordered::Bool = false) where {L <: NamedTuple, M <: Matrix}
        @argcheck typeof(values(schema)) <: Tuple{Vararg{ColumnLayout}}
        @argcheck is_valid_schema(schema, size(sample_matrix, 2))
        @argcheck thinning ≥ 1
        @argcheck warmup ≥ 0
        new{L,M}(schema, sample_matrix, thinning, warmup, is_ordered)
    end
end

"""
$(SIGNATURES)

A `NamedTuple` of `ColumnLayout`s for views into the sample matrix.
"""
schema(chain::Chain) = chain.schema

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

"""
$(SIGNATURES)

Return a generator that returns named tuples from the posterior (not including the warmup).
"""
function posterior(c::Chain)
    sch = schema(c)
    (map(layout -> view(row, layout), sch) for row in eachrow(sample_matrix(c)))
end

end
