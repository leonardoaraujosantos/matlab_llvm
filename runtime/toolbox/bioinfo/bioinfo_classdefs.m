% Bioinformatics Toolbox — Tier-4 classdef umbrella.
% Auto-prepended by matlabc when the user input mentions `phytree`,
% `seqlinkage`, or `seqneighjoin`.  The phylogenetic-tree object is a thin
% descriptor; all numeric work (UPGMA / neighbor-joining tree building,
% Newick serialization, patristic distances) lives in
% runtime/toolbox/bioinfo/runtime_bioinfo.cpp.
%
% seqlinkage / seqneighjoin are NOT classdef methods — they are registered
% builtins dispatched in Lowering.cpp that allocate this `phytree` shell then
% call the runtime populate step (matlab_bioinfo_seqlinkage*/_seqneighjoin*).
% getnewickstr / pdist / get key on the object's class at runtime (REPL-safe),
% reading the fields below.
%
% Fields: NumLeaves (N); Pointers ((N-1)x2 child node-id matrix, leaves
% 1..N, internal nodes N+1..2N-1); Distances (edge length to parent per node
% id, column vector); Names (newline-joined leaf names); Newick (cached
% Newick string).

classdef phytree
    properties
        NumLeaves
        Pointers matrix
        Distances matrix
        Names
        Newick
    end
    methods
        function obj = phytree()
            obj.NumLeaves = 0;
            obj.Pointers  = zeros(1, 2);
            obj.Distances = zeros(1, 1);
            obj.Names     = 0;
            obj.Newick    = 0;
        end
        function s = getnewickstr(obj)
            s = matlab_bioinfo_phytree_newick(obj);
        end
        function d = pdist(obj)
            d = matlab_bioinfo_phytree_pdist(obj);
        end
        function v = get(obj, prop)
            v = matlab_bioinfo_phytree_get(obj, prop);
        end
    end
end

% DataMatrix — Tier-6 microarray data container (a thin labelled wrapper over
% a numeric matrix; the normalization / filtering / clustering functions take
% the plain `.Data` matrix).  Pure-.m constructor: stores the matrix + its
% dimensions; the gene/sample-name labels round-trip through the runtime
% string-field path.
classdef DataMatrix
    properties
        Data matrix
        NRows
        NCols
    end
    methods
        function obj = DataMatrix(data)
            if nargin < 1, data = zeros(0, 0); end
            obj.Data  = data;
            obj.NRows = size(data, 1);
            obj.NCols = size(data, 2);
        end
    end
end
