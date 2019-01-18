using MCMCStorage, Test

using MCMCStorage.StanCSV: parse_variable_name, collapse_contiguous_dimensions,
   parse_schema, read_chain, chain_id_files

####
#### MCMCChains
####

@testset "schema, layout, views" begin
    named_dims = (a = (), b = (1, 2), c = (2, 3, 4))
    s = Chains.ColumnSchema(named_dims)
    l = mapreduce(prod, +, named_dims)
    @test length(s) == l
    v = 1:l
    m = reshape(v, 1, :)
    @test view(v, s.a) == 1
    @test view(v, s.b) == reshape(2:3, 1, 2)
    @test view(v, s.c) == reshape(4:l, 2, 3, 4)
    @test view(v, s) == (a = view(v, s.a), b = view(v, s.b), c = view(v, s.c))
    @test view(m, s.a) == [1]
    @test view(m, s.b) == reshape(2:3, 1, 1, 2)
    @test view(m, s.c) == reshape(4:l, 1, 2, 3, 4)
    @test view(m, s) == (a = view(m, s.a), b = view(m, s.b), c = view(m, s.c))
end

@testset "chains" begin
    sample = Float64.(hcat(1:10, 2:2:20, 3:3:30))
    sch = Chains.ColumnSchema((a = (), b = (2, )))
    chain = Chains.Chain(sch, sample; warmup = 3, is_ordered = true, thinning = 2)
    @test Chains.sample_matrix(chain) == sample[4:end, :]
    @test Chains.sample_matrix(chain, Val(true)) == sample
    @test Chains.thinning(chain) == 2
    @test Chains.warmup(chain) == 3
    p = collect(Chains.posterior(chain))
    @test p == [(f = Float64(i); (a = i, b = [2*i, 3*i])) for i in 4:10]
    c2 = vcat(chain, chain)
    @test Chains.sample_matrix(c2) == vcat(sample[4:end, :], sample[4:end, :])
    @test Chains.warmup(c2) == 0
    @test Chains.thinning(c2) == nothing
end

####
#### StanCSV
####

@testset "variable name parsing" begin
    @test parse_variable_name("a") == (:a => ())
    @test parse_variable_name("kappa.1.3") == (:kappa => (1, 3))
    @test parse_variable_name("stepsize__") == (:stepsize__ => ())
    @test_throws ArgumentError parse_variable_name("b.")
    @test_throws ArgumentError parse_variable_name("b.12.")
    @test_throws ArgumentError parse_variable_name("")
end

@testset "collapsing" begin
    v = [:a => (1, 1), :a => (2, 1), :a => (1, 2), :a => (2, 2)]
    @test collapse_contiguous_dimensions(v, 1) == (:a, (2, 2), 5)
    @test_throws ArgumentError collapse_contiguous_dimensions(v, 2)
    @test_throws ArgumentError collapse_contiguous_dimensions(v[1:(end-1)], 1)
    @test_throws ArgumentError collapse_contiguous_dimensions(v[[1, 2, 4]], 1)
end

function make_parsed_header(name_size_pairs)
    v = Any[]
    for (name, dims) in name_size_pairs
        append!(v, collect(name => Tuple(ix) for ix in CartesianIndices(dims)))
    end
    v
end

@testset "parsing schema" begin
    v = [:a => (), :b => (1, 2), :c => (3, 4, 7), :d => ()]
    @test parse_schema(make_parsed_header(v)) == v
end

@testset "reading CSV data" begin

    "Print `n` comment lines to `io`."
    function comment!(io::IO, n = 1)
        for _ in 1:n
            println(io, "# comment")
        end
    end

    # generate test data
    io = IOBuffer()
    comment!(io, rand(1:5))
    println(io, "a, b.1, b.2, c.1.1, c.2.1, c.1.2, c.2.2")
    comment!(io, rand(1:3))
    for i in 1:10
        f = Float64(i)
        print(io, f, ", ", f + 1, " ,", f + 2) # space deliberately mix in spaces
        for j in 1:4
            print(io, ",",  i + j + 4.0) # c[:, :]
        end
        println(io)
        rand() < 0.5 && comment!(io) # intersperse with comments
    end
    comment!(io, rand(1:4))
    contents = String(take!(io))

    # read test data
    name_array_pairs = read_chain(IOBuffer(contents))
    @test name_array_pairs[1] == (:a => Float64.(1:10))
    @test name_array_pairs[2] == (:b => Float64.(hcat(2:11, 3:12)))
    @test first(name_array_pairs[3]) == :c
    c = last(name_array_pairs[3])
    for i in 1:10
        @test c[i, :, :] == reshape(1:4, 2, 2) .+ 4.0 .+ i
    end
end

@testset "filename discovery" begin
    mktempdir() do dir
        base = "foo_"
        id_files = [i => joinpath(dir, base * string(i) * ".csv") for i in 1:10]
        for (_, file) in id_files
            touch(file)
        end
        @test chain_id_files(joinpath(dir, base)) == id_files
    end
end
