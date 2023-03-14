
Below you will find the companing files for the *Clustering via hypergraph modularity* 
submitted to the PLOS ONE journal by Bogumił Kamiński, Valérie Poulin,
Paweł Prałat, Przemysław Szufel and François Théberge.

In the paper, a hypergraph modularity function is proposed that 
generalizes its well established and widely used graph 
counterpart measure of how clustered a network is.

The provided code files contain implementation of heuristic algorithm 
presented in the paper along with the following two numerical examples:

- [Testing heuristics on synthetic hypergraphs](#sec_synthetic) (Subsection 4.1 of the paper)
- [Implementation of the CNM-like algorithm for hypergraphs and numerical experiments](#sec_cnm)(Subsection 4.2 of the paper)
- [Testing heuristics on the DBLP dataset represented as a hypergraph](#sec_dblp) (Subsection 4.3 of the paper)

The code that has beem used for experiments is acompanied with the 
[data files](https://gist.github.com/pszufe/02666497d2c138d1b2de5b7f67784d2b/raw/f281cfb6320023292c4b2e403a86c7e0eead5350/zipped-data-files-Clustering_via_hypergraph_modularity.zip)
that are required to run our examples. 
In order to ensure a full replicability of our research,
in the source code a fixed random number seed has been used. 

## Installation and configuration.

In order to run the examples we recommend using Python version 3
that can be downloaded via the
[Python Anaconda](https://www.anaconda.com/distribution/) distribution.
Once your Anaconda is installed, you need to install the `igraph` package,
simply execute tha bash command below:

```bash
conda install -c vtraag python-igraph
```


## <a name="sec_synthetic"></a> Test on synthetic hypergraphs

In this numerical example, we generate hyperedges 
following the process in M. Leodeanu et al. (2012).

We consider 150 points on the space [-.5,.5]x[-.5,.5] such that:

Nodes 0-29:  noisy points on a line A with slope -1
Nodes 30-59: noisy points on a line B with slope 0.02
Nodes 60-99: noisy points on a line C with slope 0.8
Nodes 100-159: random points

We sample points in sets of 3 and 4 respecteivly, keeping
groups of points that are well-aligned (w.r.t. residuals
of the numpy.polyfit() function).

The hyperedges are listed in the [supplied files](https://gist.github.com/pszufe/02666497d2c138d1b2de5b7f67784d2b/raw/f281cfb6320023292c4b2e403a86c7e0eead5350/zipped-data-files-Clustering_via_hypergraph_modularity.zip):

  hypergraph_3uniform.csv: 159,816 3-edges
  hypergraph_4uniform.csv: 160,000 4-edges

To generate our plots, we randomly sampled from those sets
with respect to 3 different regimes:

- same proportion of 3 and 4-edges
- 2/3 3-edges, or
- 2/3 4-edges

and we also sample such that we expect twice as many edges 
all coming from the same line (A, B or C) as we have sets with 
mixed lines and/or random points.

To generate the plots, just run:

```bash
	python hypergraph.py
```
This requires the functions supplied in the strictModularity.py file.
Hyperedges are read from the 2 csv files.
Two plots are produced: `lines_qG.eps` and `lines_qH.eps`. They constitute
the Figure 1 in our paper.   
In order to ensure replicability of our research we use a fixed random 
number seed. However, you are free to change it in order to see 
that the results may vary slightly with each sampling but the conclusions
are stable.

## <a name="sec_cnm"></a> Implmentation and test of CM-like algorithm
The code works with the file generated in the previous step

```julia
using SimpleHypergraphs, StatsBase

function do_file(name::String)
    f = open(name)
    line= readline(f)
    h = Hypergraph{Bool,Int}(0,0)

    for v_meta in parse.(Int,(split(line,"\t")))
        add_vertex!(h,vertex_meta=v_meta)
    end
    for line in eachline(f)
        x = parse.(Int,(split(line,"\t")))
        inds = x .+ 1
        add_hyperedge!(h;vertices=Dict(inds .=> true))

    end
    close(f)
    h
end
function find_first(c::Array{Set{Int}}, vals)
    for i in 1:length(c)
        for v in vals
            v in c[i] && return i
        end
    end
end

function find_comms(h::Hypergraph, nreps::Int=1000; ha = SimpleHypergraphs.HypergraphAggs(h))
    best_modularity = 0
    comms = [Set(i) for i in 1:nhv(h)]
    mod_history = Vector{Float64}(undef, nreps)
    for rep in 1:nreps
        he = rand(1:nhe(h))
        vers = collect(keys(getvertices(h, he)))
        c = deepcopy(comms)
        i0 = find_first(c, vers)
        max_i = length(c)
        i_cur = i0
        while i_cur < max_i
            i_cur += 1
            if length(intersect(c[i_cur],vers)) > 0
                union!(c[i0], c[i_cur])
                c[i_cur]=c[max_i]
                max_i += -1
            end
        end
        resize!(c,max_i)
        m = modularity(h, c, ha)
        if m > best_modularity
            best_modularity = m
            comms = c
        end
        mod_history[rep] = best_modularity
    end
    return (bm=best_modularity, bp=comms, mod_history=mod_history)
end
```

Now let use the code to perform experiments:
```julia
const h = do_file("h2_100");
const ha = SimpleHypergraphs.HypergraphAggs(h)
res = Matrix{Float64}(undef,500,0)
for i=1:1920  #in the production code we use distributed computing here.
    res = hcat(res, find_comms(h,500, ha=ha)[3])
end
using DelimitedFiles
writedlm("res.txt",res)
```

Once the above code has been run we have used the data to create the Figure 2. 




## <a name="sec_dblp"></a> Test on a small DBLP dataset

The DBLP computer science bibliography database contains 
open bibliographic information on major computer science 
journals and proceedings. 
The [DBLP database](http://dblp.uni-trier.de/xml/) is operated jointly by University of 
Trier and Schloss Dagstuhl. 
We consider a hypergraph of citations where each 
vertex represents an author and hyperedges are papers.

The data files for this experiment are available 
[here](https://gist.github.com/pszufe/02666497d2c138d1b2de5b7f67784d2b/raw/f281cfb6320023292c4b2e403a86c7e0eead5350/zipped-data-files-Clustering_via_hypergraph_modularity.zip)
and contain preprocessed DBLP data.
The file `dblp_edges` contains hyper-edges, where
each line is of the form {2, 3, 4, 5}; 
the digits correspond to unique authors;
for each author, we have identified a field of academic
interest, which are listed in the file `dblp_authors-fields`

As reported in the paper, we pruned all edges that contained 
at least one author with `unknown` field of interest 
and all edges of size 1; 
we also randomly selected 1/3 of the edges, 
and we kept only the (unique) large connected component.

We ended up with 1637 nodes, 865 edges of size 2, 470 of size 3,
152 of size 4 and 37 of size 5 to 7.

Those are included in the file `dblp.pkl` along with author
fields and the partitions we obtained.

To re-create the results from the paper, run

```bash
	python dblp.py
```

By default, the partitions are read from the `dblp.pkl` file;
un-comment the lines in the Python code to re-run the Louvain
and hypergraph-CNM algorithms (this one is slow).

This code is also using the basic functions that are loaded 
from the `strictModularity.py` file. Output is sent to `stdout`.

## Licensing
This file and all supplied [datasets](https://gist.github.com/pszufe/02666497d2c138d1b2de5b7f67784d2b/raw/f281cfb6320023292c4b2e403a86c7e0eead5350/zipped-data-files-Clustering_via_hypergraph_modularity.zip) are licensed under the terms of Creative Commons ["By Attribution" License 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

All source code in this archive is licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT). 