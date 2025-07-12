import os, sys, argparse, time
from multiprocessing import Pool
from multicom3.common.util import check_file, check_dir, check_dirs, makedir_if_not_exists, check_contents, \
    read_option_file
from multicom3.monomer_alignment_generation.alignment import write_fasta
from multicom3.common.protein import read_qa_txt_as_df, parse_fasta, complete_result, make_chain_id_map
from multicom3.quaternary_structure_refinement import iterative_refine_pipeline_multimer
from multicom3.monomer_structure_refinement import iterative_refine_pipeline
from multicom3.common.pipeline import run_monomer_msa_pipeline, run_monomer_template_search_pipeline, \
    run_monomer_structure_generation_pipeline_v2, run_monomer_evaluation_pipeline, run_monomer_refinement_pipeline, \
    run_concatenate_dimer_msas_pipeline, run_complex_template_search_pipeline, \
    run_quaternary_structure_generation_pipeline_v2, \
    run_quaternary_structure_generation_pipeline_foldseek, run_multimer_refinement_pipeline, \
    run_multimer_evaluation_pipeline, run_monomer_msa_pipeline_img, foldseek_iterative_monomer_input, \
    copy_same_sequence_msas

from absl import flags
from absl import app
import copy
import pandas as pd
import subprocess
import time
import datetime

flags.DEFINE_string('option_file', None, 'option file')
flags.DEFINE_string('fasta_path', None, 'Path to multimer fasta')
flags.DEFINE_string('output_dir', None, 'Output directory')
flags.DEFINE_boolean('run_img', False, 'Whether to use IMG alignment to generate models')
FLAGS = flags.FLAGS


def main(argv):
    print("#################################################################################################")
    print(argv)

    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '4.0'

    check_file(FLAGS.option_file)

    params = read_option_file(FLAGS.option_file)

    makedir_if_not_exists(FLAGS.output_dir)

    check_dirs(params, ['hhblits_program', 'jackhmmer_program'], isdir=False)

    check_file(FLAGS.fasta_path)

    makedir_if_not_exists(FLAGS.output_dir)

    N1_outdir = FLAGS.output_dir + '/N1_monomer_alignments_generation'
    Time_file = FLAGS.output_dir + '/execution_time.txt'

    print("#################################################################################################")

    print("#################################################################################################")
    print("1-3. Start to generate monomer models")

    makedir_if_not_exists(N1_outdir)

    with open(FLAGS.fasta_path) as f:
        input_fasta_str = f.read()
    input_seqs, input_descs = parse_fasta(input_fasta_str)
    chain_id_map, chain_id_seq_map = make_chain_id_map(sequences=input_seqs,
                                                       descriptions=input_descs)

    print("#################################################################################################")

    print("#################################################################################################")
    print("4. Start to generate complex alignments")
    N4_outdir = FLAGS.output_dir + '/N4_complex_alignments_concatenation'
    makedir_if_not_exists(N4_outdir)
    start_time = time.time()
    try:
        concat_methods = ['pdb_interact', 'species_interact', 'uniclust_oxmatch',
                           'string_interact', 'uniprot_distance', 'PLM_interact', 'PLM_interact_structure_sim']
        run_concatenate_dimer_msas_pipeline(
            multimer=','.join([chain_id_map[chain_id].description for chain_id in chain_id_map]),
            run_methods=concat_methods,
            monomer_aln_dir=N1_outdir, outputdir=N4_outdir, params=params)
    except Exception as e:
        print(e)
        print("Program failed in step 5")
    end_time = time.time()
    execution_time = end_time - start_time
    with open(Time_file, 'a') as file:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f'({current_time}) Time N4: {execution_time} seconds\n')

    print("#################################################################################################")

if __name__ == '__main__':
    flags.mark_flags_as_required([
        'option_file',
        'fasta_path',
        'output_dir'
    ])
    app.run(main)
