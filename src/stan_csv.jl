"""
Read CSV-formatted posterior output.

# API

See [`StanCSV.read_chain`](@ref) and [`StanCSV.matching_files`], neither is exported.

# File format

Posterior samples are supposed to be available in a CSV file which follows the following
conventions:

1. Whitespace and content after `#` is ignored. No fields are quoted or escaped. If a line
has a `#`, it cannot have content.

2. The first non-ignored line contains a comma-separated list of variable names, of the
format `var[.i1.i2.…]` where optional indexes follow dots. Indexes for the same variable are
contiguous, and have a column-major layout.

3. Subsequent lines contain the same number of comma-separated floating-point values.

`cmdstan` outputs posterior samples in this format (diagnostics are ignored for now), hence
the name.
"""
module StanCSV

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES
import ..Chains

####
#### Header parsing building blocks
####

"""
$(SIGNATURES)

Parse a Stan variable name as `name::Symbol => indexes::Tuple`.
"""
function parse_variable_name(variable_name::AbstractString)
    parts = split(variable_name, '.')
    @argcheck !isempty(first(parts))
    name = Symbol(strip(first(parts)))
    index = tuple(parse.(Ref(Int), parts[2:end])...)
    name => index
end

"""
$(SIGNATURES)

Given a vector of `name => index` pairs and a starting index `i`, which should be `1`s, find
the last index for this variable (ie the dimensions) and check that that intermediate
indices are contiguous (in column major ordering).

Return `name`, `dimensions`, and the position for the continuation.
"""
function collapse_contiguous_dimensions(names_indexes, i)
    name, index = names_indexes[i]
    @argcheck all(index .== 1) "Indexes for $(name) don't start with ones."
    n = length(names_indexes)
    j = i + 1
    while j ≤ n
        if names_indexes[j][1] ≠ name
            break
        else
            j += 1
        end
    end
    dimensions = names_indexes[j - 1][2]
    indexes = CartesianIndices(dimensions)
    for (k, ix) in zip(i:(j-1), CartesianIndices(dimensions))
        index_k = names_indexes[k][2]
        @argcheck CartesianIndex(index_k) == ix "Non-contiguous index $(k) for $(name)."
    end
    name, dimensions, j
end

"""
$(SIGNATURES)

Parse a vector of `names => indexes` to a column schema.
"""
function parse_schema(names_indexes)
    n = length(names_indexes)
    i = 1
    schema = Pair{Symbol,Tuple{Vararg{Int}}}[]
    while i ≤ n
        name, dimensions, next_i = collapse_contiguous_dimensions(names_indexes, i)
        @argcheck all(s -> first(schema) ≠ name, schema) "Duplicate name $(name) in column $(i)."
        push!(schema, name => dimensions)
        i = next_i
    end
    Chains.ColumnSchema(NamedTuple{Tuple(first.(schema))}(last.(schema)))
end

####
#### CSV file reading.
####

"""
$(SIGNATURES)

Test if `line` is a comment line.
"""
is_comment_line(line) = occursin(r"^ *#", line)

"""
$(SIGNATURES)

Find the first non-comment line and read it as a `Chains.ColumnSchema`. When there is no
such line, throw an `EOFError`.
"""
function read_schema(io::IO)
    while true                  # will throw an EOFError if not fount
        line = readline(io; keep = false)
        if !is_comment_line(line)
            return parse_schema(parse_variable_name.(split(line, ',')))
        end
    end
end

"""
$(SIGNATURES)

Read `nfields` fields from `io`, parse as `Float64`, and push into `buffer`, returning
`true`. Fewer fields than `nfields` results in an error.

If a line starts with a `'#'`, do no parsing, return `false`.
"""
function read_csv_line!(buffer, io::IO, nfields::Integer)
    line = readline(io; keep = false)
    is_comment_line(line) && return false
    pos = 1
    last_pos = lastindex(line)
    for _ in 1:nfields
        @argcheck pos ≤ last_pos "Fewer than $(nfields) fields in line."
        delim_pos = something(findnext(isequal(','), line, pos), last_pos + 1)
        push!(buffer, parse(Float64, SubString(line, pos, delim_pos - 1)))
        pos = delim_pos + 1
    end
    true
end

function read_csv_flat_data!(buffer, io, nfields)
    row_count = 0
    while !eof(io)
        row_count += read_csv_line!(buffer, io, nfields)
    end
    row_count
end

function csv_buffer_to_array(buffer, row_count, dims)
    n = length(dims)
    permutedims(reshape(buffer, dims..., row_count), vcat([1 + n], 1:n))
end

function read_chain(io::IO)
    sch = read_schema(io)
    nfields = length(sch)
    buffer = Vector{Float64}()
    row_count = read_csv_flat_data!(buffer, io, nfields)
    sample_matrix = collect(permutedims(reshape(buffer, nfields, row_count)))
    Chains.Chain(sch, sample_matrix)
end

"""
$(SIGNATURES)

Read data from a CSV file, that uses the output format from Stan.

Return a vector of `name => array` pairs, where `name::Symbol` is the variable name, and the
`array` is the data for that variable.

Consecutive columns for the same variable name with `n`-dimensional indexes are assembled
into an `n+1` dimensional array, with index `var[row, i1, i2, …]` containing the results
for `var.i1.i2.…` in the given `row`.
"""
read_chain(filename::AbstractString) = open(read_chain, filename, "r")

"""
$(SIGNATURES)

Return a vector `id::Int => filename` where `filename` is `prefix_<digits>.csv` and the path
is a file. Prefix should be a path, eg

```julia
$(FUNCTIONNAME)("/tmp/samples_")
```

will return `[1 => "/tmp/samples_1.csv", 2 => "/tmp/samples_2.csv", …]` if the files exist.

Recommended use: obtaining all the filenames in a directory.
"""
function matching_files(prefix::AbstractString)
    dir = dirname(prefix)
    base = basename(prefix)
    pattern = Regex("^" * base * raw"(\d+)\.csv$")
    ids_files = Pair{Int,String}[]
    for filename in readdir(dir)
        m = match(pattern, filename)
        p = joinpath(dir, filename)
        if m ≢ nothing && isfile(p)
            push!(ids_files, parse(Int, m.captures[1]) => p)
        end
    end
    @argcheck allunique(first.(ids_files)) "Non-unique file ids, perhaps because of 0-padding?"
    sort!(ids_files, by = first)
    ids_files
end

end
