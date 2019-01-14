#####
#####
#####

module StanCSV

export read_chain

using ArgCheck: @argcheck
using DocStringExtensions: SIGNATURES

function parse_variable_name(variable_name::AbstractString)
    parts = split(variable_name, '.')
    @argcheck !isempty(first(parts))
    name = Symbol(strip(first(parts)))
    index = tuple(parse.(Ref(Int), parts[2:end])...)
    name => index
end

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
    schema
end

function read_schema(io::IO)
    while true                  # will throw an EOFError if not fount
        line = readline(io; keep = false)
        if !occursin(r"^ *#", line)
            return parse_schema(parse_variable_name.(split(line, ',')))
        end
    end
end

function read_csv_line!(buffers, io::IO, contiguous_lengths)
    line = readline(io; keep = true) # treat \n as the last delimiter
    first(line) == '#' && return false
    pos = 1
    lastpos = lastindex(line)
    for (i, l) in enumerate(contiguous_lengths)
        buffer = buffers[i]
        for _ in 1:l
            delim_pos = something(findnext(isequal(','), line, pos), lastpos)
            push!(buffer, parse(Float64, SubString(line, pos, delim_pos - 1)))
            pos = delim_pos + 1
        end
    end
    true
end

function read_csv_flat_data!(buffers, io, contiguous_lengths)
    row_count = 0
    while !eof(io)
        row_count += read_csv_line!(buffers, io, contiguous_lengths)
    end
    row_count
end

function csv_buffer_to_array(buffer, row_count, dims)
    n = length(dims)
    permutedims(reshape(buffer, dims..., row_count), vcat([1 + n], 1:n))
end

function read_chain(io::IO)
    schema = read_schema(io)
    contiguous_lengths = [prod(last(s)) for s in schema]
    buffers = [Vector{Float64}() for _ in eachindex(contiguous_lengths)]
    row_count = read_csv_flat_data!(buffers, io, contiguous_lengths)
    [name => csv_buffer_to_array(buffer, row_count, dims)
     for ((name, dims), buffer) in zip(schema, buffers)]
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
read_chain(filename::AbstractString) = open(read_csv_data, filename, "r")

"""
$(SIGNATURES)

Return a vector `id::Int => filename` where `filename` is `prefix_<digits>.csv` and the path
is a file. Prefix should be a path, eg

```julia
chain_id_files("/tmp/samples_")
```

will return `[1 => "/tmp/samples_1.csv", 2 => "/tmp/samples_2.csv", …]` if the files exist.

Recommended use: obtaining all the filenames in a directory.
"""
function chain_id_files(prefix::AbstractString)
    dir = dirname(prefix)
    base = basename(prefix)
    pattern = Regex("^" * base * raw"(\d+)\.csv$")
    id_files = Pair{Int,String}[]
    for filename in readdir(dir)
        m = match(pattern, filename)
        p = joinpath(dir, filename)
        if m ≢ nothing && isfile(p)
            push!(id_files, parse(Int, m.captures[1]) => p)
        end
    end
    @argcheck allunique(first.(id_files)) "Non-unique file ids, perhaps because of 0-padding?"
    sort!(id_files, by = first)
    id_files
end

end
