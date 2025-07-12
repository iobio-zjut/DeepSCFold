import os
import argparse
import subprocess
import pandas as pd
import csv


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")


def add_query_to_a3m(a3m_file_path, query):
    if not os.path.exists(a3m_file_path):
        raise FileNotFoundError(f"The file {a3m_file_path} does not exist.")

    with open(a3m_file_path, 'r') as f:
        original_content = f.read()

    new_content = query + '\n' + original_content

    with open(a3m_file_path, 'w') as f:
        f.write(new_content)

    print(f"Query sequence added to {a3m_file_path} successfully.")

def read_fasta_sequences(file_path):
    sequences = []
    current_sequence = []
    query = None
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    sequence = '\n'.join(current_sequence)
                    if query is None:
                        query = sequence
                    else:
                        sequences.append(sequence)
                current_sequence = [line]
            else:
                cleaned_line = ''.join(
                    '-' if char == 'X' or char == 'U' else (
                        char if char.isupper() and char not in ['X', 'U'] or char.isspace() or char == "-" else '')
                    for char in line
                )
                current_sequence.append(cleaned_line)
        if current_sequence:
            sequence = '\n'.join(current_sequence)
            if query is None:
                query = sequence
            else:
                sequences.append(sequence)
    return query, sequences


def save_sequences_to_files(sequences_1, sequences_2, output_dir, prefix_1, prefix_2, n_files):
    sequences_per_file = len(sequences_1) // n_files
    output_files = []
    for i in range(n_files):
        start_index = i * sequences_per_file
        end_index = (i + 1) * sequences_per_file if i < n_files - 1 else len(sequences_1)
        output_file_1 = os.path.join(output_dir, f"{prefix_1}_part_{i + 1}.a3m")
        output_file_2 = os.path.join(output_dir, f"{prefix_2}_part_{i + 1}.a3m")
        with open(output_file_1, 'w') as file_1, open(output_file_2, 'w') as file_2:
            for seq in sequences_1[start_index:end_index]:
                file_1.write(f"{seq}\n")
            for seq in sequences_2[start_index:end_index]:
                file_2.write(f"{seq}\n")
        output_files.append((output_file_1, output_file_2))
    return output_files


def run_prediction(output_files, script_path, output_dir):
    index = 0
    for output_file_1, output_file_2 in output_files:
        index += 1
        output_dir_part = os.path.join(output_dir, str(index))
        os.makedirs(output_dir_part, exist_ok=True)
        command = f"python {script_path} --query_path {output_file_1} --target_path {output_file_2} --output_dir {output_dir_part}"
        print(f"Running command: {command}")
        subprocess.run(command, shell=True, check=True)


def combine_final_scores(output_dir, output_file):
    combined_scores = []
    for subdir in sorted(os.listdir(output_dir)):
        subdir_path = os.path.join(output_dir, subdir)
        if os.path.isdir(subdir_path):
            final_scores_path = os.path.join(subdir_path, 'final_scores.csv')
            if os.path.exists(final_scores_path):
                df = pd.read_csv(final_scores_path)
                combined_scores.append(df)

    if combined_scores:
        combined_df = pd.concat(combined_scores)
        combined_df.to_csv(output_file, index=False)
        print(f"Combined final scores saved to {output_file}")
    else:
        print("No final_scores.csv files found to combine.")


def pair_MSA_generation(script_path, input_score_csv, input_fasta_1, input_fasta_2, output_pairs_csv, output_fasta_1,
                        output_fasta_2, output_complex_msa):
    command = f"python {script_path} --input_score_csv {input_score_csv} --input_msa_for_chain1 {input_fasta_1} --input_msa_for_chain2 {input_fasta_2} --output_pairs_csv {output_pairs_csv} --output_msa_for_chain1 {output_fasta_1} --output_msa_for_chain2 {output_fasta_2} --output_complex_msa {output_complex_msa}"
    print(f"Running command: {command}")
    subprocess.run(command, shell=True, check=True)


def extract_header_and_sequence(fasta_entry):
    lines = fasta_entry.split('\n')
    header = lines[0].lstrip('>')
    sequence = ''.join(lines[1:])
    return header, sequence


def create_combined_query(query1, query2):
    header1, seq1 = extract_header_and_sequence(query1)
    header2, seq2 = extract_header_and_sequence(query2)

    combined_header = f">{header1}_____{header2}"
    combined_sequence = seq1 + seq2

    return f"{combined_header}\n{combined_sequence}"


def write_combined_msa(output_file, combined_query):
    with open(output_file, 'w') as f:
        f.write(combined_query)
    print(f"Combined query written to {output_file} successfully.")


def insert_headers_to_csv(csv_file_path, header1, header2):
    with open(csv_file_path, 'r') as csv_file:
        reader = list(csv.reader(csv_file))

    new_row = [header1, '0', header2, '0', '1']
    reader.insert(1, new_row)

    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(reader)

    print(f"Headers inserted to {csv_file_path} successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and split sequences from input FASTA files.')
    # input
    parser.add_argument('--input_msa_1', type=str, required=True, help='Path to the first input FASTA file.')
    parser.add_argument('--input_msa_2', type=str, required=True, help='Path to the second input FASTA file.')

    # output
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files.')

    parser.add_argument('--output_pairs_csv', type=str, required=False, default='output_pairs.csv',
                        help='The name to hold the output CSV file. Default is "output_pairs.csv".')
    parser.add_argument('--output_msa_for_chain1', type=str, required=False, default='ChainA_con.a3m',
                        help='The name to hold the MSA file extracted sequences from the first input file. Default is "ChainA_con.a3m".')
    parser.add_argument('--output_msa_for_chain2', type=str, required=False, default='ChainB_con.a3m',
                        help='The name to hold the MSA file extracted sequences from the second input file. Default is "ChainB_con.a3m".')
    parser.add_argument('--output_complex_msa', type=str, required=False, default='output_complex_msa.msa',
                        help='File name of the output pair-msa. Default is "output_complex_msa.a3m".')

    # others
    parser.add_argument('--seqs_per_file', type=int, default=100, help='Number of sequences per output file.')
    parser.add_argument('--score_script_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'predict_for_pIA_score.py'),
                        help='Path to the predict_for_pIA_score.py script.')
    parser.add_argument('--msa_generation_script_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'pair_msa_generation.py'),
                        help='Path to the pair_msa_generation.py script.')

    args = parser.parse_args()

    ensure_directory_exists(args.output_dir)

    query1, sequences1 = read_fasta_sequences(args.input_msa_1)
    query2, sequences2 = read_fasta_sequences(args.input_msa_2)

    # set number
    num_sequences = min(len(sequences1), len(sequences2), 512)
    n_files = max(1, (num_sequences + args.seqs_per_file - 1) // args.seqs_per_file)
    sequences1 = sequences1[:num_sequences]
    sequences2 = sequences2[:num_sequences]

    # save sequences
    output_files = save_sequences_to_files(sequences1, sequences2, args.output_dir, "chain1", "chain2", n_files)

    # predict pIA-score
    run_prediction(output_files, args.score_script_path, args.output_dir)

    # generate final_scores.csv
    combined_score_file = os.path.join(args.output_dir, 'final_interact_score.csv')
    combine_final_scores(args.output_dir, combined_score_file)

    # input path
    path_output_pairs_csv = os.path.join(args.output_dir, args.output_pairs_csv)
    path_output_msa_for_chain1 = os.path.join(args.output_dir, args.output_msa_for_chain1)
    path_output_msa_for_chain2 = os.path.join(args.output_dir, args.output_msa_for_chain2)
    path_output_pair_msa = os.path.join(args.output_dir, args.output_complex_msa)

    # pair-MSA
    pair_MSA_generation(args.msa_generation_script_path, combined_score_file, args.input_msa_1, args.input_msa_2,
                        path_output_pairs_csv, path_output_msa_for_chain1, path_output_msa_for_chain2,
                        path_output_pair_msa)

    # monomer msa file generation
    add_query_to_a3m(path_output_msa_for_chain1, query1)
    add_query_to_a3m(path_output_msa_for_chain2, query2)

    # multimer msa file generation
    combined_query = create_combined_query(query1, query2)
    add_query_to_a3m(path_output_pair_msa, combined_query)

    # index_csv file generation
    header1, _ = extract_header_and_sequence(query1)
    header2, _ = extract_header_and_sequence(query2)
    insert_headers_to_csv(path_output_pairs_csv, header1, header2)
