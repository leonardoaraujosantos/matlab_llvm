# Bioinformatics Toolbox Spec

## Purpose
Document the shipped subset of MATLAB's Bioinformatics Toolbox in `matlab_llvm`: sequence I/O and manipulation, pairwise/multiple alignment, phylogenetics, microarray normalization/filtering, and mass-spectrometry preprocessing. Built mostly over char-vector string algorithms and shipped numeric kernels.

## Requirements

### Requirement: Sequence I/O and conversion
The system SHALL read/write FASTA sequences and convert between nucleotide/amino-acid alphabets.

#### Scenario: Read a FASTA and convert sequences
- **WHEN** a program calls `fastaread`/`fastawrite`, `dna2rna`/`rna2dna`, `nt2aa`, `seqcomplement`/`seqrcomplement`, `seqreverse`, `randseq`, or alphabet integer encodings (`nt2int`/`aa2int`)
- **THEN** the system SHALL return the parsed sequences or converted sequence (matlab_bioinfo_fastaread, matlab_bioinfo_dna2rna, matlab_bioinfo_nt2aa, matlab_bioinfo_seqrcomplement, matlab_bioinfo_randseq) (doc: docs/bioinformatics_toolbox_roadmap.md) (src: runtime/toolbox/bioinfo/runtime_bioinfo.cpp, runtime/toolbox/bioinfo/bioinfo_classdefs.m)

### Requirement: Sequence statistics and composition
The system SHALL compute sequence composition and biochemical properties.

#### Scenario: Compute composition and properties
- **WHEN** a program calls `basecount`/`aacount`, `molweight`, `isoelectric`, `atomiccomp`, `aminolookup`, or restriction `cleave`/`restrict`
- **THEN** the system SHALL return the composition counts or computed property (matlab_bioinfo_basecount, matlab_bioinfo_aacount, matlab_bioinfo_molweight, matlab_bioinfo_isoelectric, matlab_bioinfo_cleave) (doc: docs/bioinformatics_toolbox_roadmap.md) (src: runtime/toolbox/bioinfo/runtime_bioinfo.cpp, runtime/toolbox/bioinfo/bioinfo_classdefs.m)

### Requirement: Sequence alignment
The system SHALL perform pairwise and multiple sequence alignment.

#### Scenario: Align two sequences
- **WHEN** a program calls Needleman-Wunsch (`nwalign`) or Smith-Waterman (`swalign`) with a scoring matrix (`blosum`/`nuc44`), or multiple alignment (`multialign`/`profalign`)
- **THEN** the system SHALL return the alignment score and aligned sequences (matlab_bioinfo_nwalign_align2, matlab_bioinfo_swalign_align2, matlab_bioinfo_blosum, matlab_bioinfo_multialign, matlab_bioinfo_profalign) (doc: docs/bioinformatics_toolbox_roadmap.md) (src: runtime/toolbox/bioinfo/runtime_bioinfo.cpp, runtime/toolbox/bioinfo/bioinfo_classdefs.m)

### Requirement: Phylogenetics
The system SHALL build and serialize phylogenetic trees.

#### Scenario: Build a phylogenetic tree
- **WHEN** a program computes `seqpdist`, builds a tree by linkage/neighbor-joining (`seqlinkage`/`seqneighjoin`), queries a `phytree`, or writes Newick (`phytreewrite`)
- **THEN** the system SHALL return the distance matrix, the tree, or the serialized Newick string (matlab_bioinfo_seqpdist2, matlab_bioinfo_seqlinkage1, matlab_bioinfo_seqneighjoin1, matlab_bioinfo_phytree_newick, matlab_bioinfo_phytreewrite) (doc: docs/bioinformatics_toolbox_roadmap.md) (src: runtime/toolbox/bioinfo/runtime_bioinfo.cpp, runtime/toolbox/bioinfo/bioinfo_classdefs.m)

### Requirement: Microarray analysis
The system SHALL normalize and filter microarray expression data.

#### Scenario: Normalize microarray data
- **WHEN** a program calls `quantilenorm`, `manorm`, gene filters (`genevarfilter`/`genelowvalfilter`/`generangefilter`), `rankfeatures`, `knnimpute`, or `crossvalind` over a `DataMatrix`
- **THEN** the system SHALL return the normalized/filtered expression data or partition (matlab_bioinfo_quantilenorm, matlab_bioinfo_manorm, matlab_bioinfo_genevarfilter, matlab_bioinfo_knnimpute, matlab_bioinfo_crossvalind) (doc: docs/bioinformatics_toolbox_roadmap.md) (src: runtime/toolbox/bioinfo/runtime_bioinfo.cpp, runtime/toolbox/bioinfo/bioinfo_classdefs.m)

### Requirement: Mass spectrometry preprocessing
The system SHALL preprocess and detect peaks in mass-spectrometry signals.

#### Scenario: Preprocess a mass spectrum
- **WHEN** a program calls `msbackadj`, `msnorm`, `msresample`, `mslowess`, or peak detection (`mspeaks`)
- **THEN** the system SHALL return the baseline-corrected/normalized signal or detected peaks (matlab_bioinfo_msbackadj, matlab_bioinfo_msnorm, matlab_bioinfo_msresample, matlab_bioinfo_mslowess, matlab_bioinfo_mspeaks) (doc: docs/bioinformatics_toolbox_roadmap.md) (src: runtime/toolbox/bioinfo/runtime_bioinfo.cpp, runtime/toolbox/bioinfo/bioinfo_classdefs.m)
