# DeepSCFold

DeepSCFold is designed for high-accuracy protein complex structure modeling using complex paired MSAs constructed by identifying potential interaction relationships between monomeric sequences. In the paired-MSA construction, the key component of DeepSCFold is two sequence-based deep learning models that predicts protein-protein structural similarity (pSS-score) and interaction probability (pIA-score). For the input protein complex sequences, DeepSCFold first generates monomeric multiple sequence alignments (MSAs) from multiple sequence databases (such as UniRef30, UniRef90, UniProt, BFD, MGnify, and the ColabFold DB). Then, the predicted pSS-score between the input sequence and each sequence alignment in the monomer MSAs was used for ranking and selecting the monomer MSAs. Subsequently, the developed deep learning model predicts the pIA-scores for the sequence alignments from different subunit MSAs to construct paired MSAs. Additionally, we use information from multiple sources, such as species annotations, UniProt accession number, and protein complexes from the Protein Data Bank (PDB), to further construct extra paired MSAs. Subsequently, DeepSCFold uses the series of paired MSAs constructed above to perform complex structure predictions through AlphaFold-Multimer. The top-1 model is selected based on our in-house complex model quality assessment method.

## **Overall workflow for the DeepSCFold Protein tertiary structure prediction system**
![DeepSCFold pipeline](Pipline.png)

# **Download DeepSCFold package**

```
git clone --recursive https://github.com/iobio-zjut/DeepSCFold 
```

# **Installation**

## **Install [AlphaFold/AlphaFold-Multimer](https://github.com/google-deepmind/alphafold), [MULTICOM3](https://github.com/BioinfoMachineLearning/MULTICOM3/releases/tag/v1.0.0) and other required third-party packages**

### **Install miniconda**

``` bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh
```

### **Create a new conda environment and update**

``` bash
conda create --name deepscfold python==3.8
conda update -n base conda
```

### **Activate conda environment**

``` bash
conda activate deepscfold
```

### **Install dependencies**

```
python>=3.8
numpy>=1.24.4
pytorch>=1.11.0
torchvision>=0.12.0
scipy>=1.7.3
pandas>=2.0.3
tqdm>=4.67.1
biopython>=1.83
```

### **Modified from [MULTICOM3](https://github.com/BioinfoMachineLearning/MULTICOM3/releases/tag/v1.0.0)**

- **Replace the following files** in `MULTICOM/multicom3/complex_alignment_generation/` with the versions from `multicom3/complex_alignment_generation/`:

  - `pipeline_v3.py`
  - `pdb_interact_v3.py`
  - `species_interact_v3.py`
  - `string_interact_v3.py`
  - `uniclust_oxmatch_v3.py`
  - `uniprot_distance_v3.py`

- **Add the following new files** from `multicom3/complex_alignment_generation/` to `MULTICOM/multicom3/complex_alignment_generation/`:

  - `plm_interact_seq_similarity_v3.py`
  - `plm_interact_stru_similarity_v3.py`

- **Replace the file** `pipeline_v2.py` in `MULTICOM/multicom3/quaternary_structure_generation/` with the one from `multicom3/quaternary_structure_generation/`.

# **Running the protein complex structure prediction pipeline**

The monomer MSA, template, and structure can be generated following the standard MULTICOM3 pipeline. Then, use the code below to generate the pair-MSA.
```
python your/path/to/multicom3/heteromer_4.py --option_file=your/path/to/MULTICOM/bin/db_option --fasta_path=your/path/to/fasta/file --output_dir=your/path/to/output --run_img=false
```
Cntinue with the remaining steps of the MULTICOM3 pipeline to generate the complex structure.

**NOTE:** For model quality assessment, please refer to **DeepUMQA-X**: [http://zhanglab-bioinf.com/DeepUMQA-X/](http://zhanglab-bioinf.com/DeepUMQA-X/)
