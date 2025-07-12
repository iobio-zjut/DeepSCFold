import os
import csv
import argparse


def read_and_sort_pairs(csv_file):
    pairs = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            query_seqid, target_seqid, score = row
            pairs.append((query_seqid, target_seqid, float(score)))

    sorted_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)

    return sorted_pairs


def filter_unique_pairs(sorted_pairs, index_1_start, index_2_start):
    unique_pairs = []
    seen_sequences = set()
    index_1 = index_1_start
    index_2 = index_2_start

    for query_seqid, target_seqid, score in sorted_pairs:
        if query_seqid not in seen_sequences and target_seqid not in seen_sequences:
            unique_pairs.append((query_seqid, index_1, target_seqid, index_2, score))
            seen_sequences.add(query_seqid)
            seen_sequences.add(target_seqid)
            index_1 += 1
            index_2 += 1

    return unique_pairs


def save_filtered_pairs_to_csv(filtered_pairs, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id_1', 'index_1', 'id_2', 'index_2', 'score'])
        for query_seqid, index_1, target_seqid, index_2, score in filtered_pairs:
            if score >= 0.5:
                writer.writerow([query_seqid, index_1, target_seqid, index_2, score])

def read_sorted_pairs(csv_file, column_index):
    ids = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            ids.append(row[column_index])
    return ids


def extract_sequences(input_file, sequence_ids):
    sequences = {}
    sequence_ids_set = set(sequence_ids)
    current_id = None
    current_sequence = []

    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_id in sequence_ids_set:
                    sequences[current_id] = ''.join(current_sequence)
                current_id = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)

        if current_id in sequence_ids_set:
            sequences[current_id] = ''.join(current_sequence)

    ordered_sequences = {seq_id: sequences[seq_id] for seq_id in sequence_ids if seq_id in sequences}
    return ordered_sequences


def save_sequences(output_file, sequences):
    with open(output_file, 'w') as file:
        for seq_id, sequence in sequences.items():
            file.write(f'>{seq_id}\n')
            file.write(f'{sequence}\n')


def complex_pair_msas_generation(chainA_msa, chainB_msa, output_msa):
    msa_A_DICT = {}
    with open(chainA_msa, 'r') as msafile_A:
        for line in msafile_A:
            line = line.strip()
            if line.startswith('>'):
                key = line
                msa_A_DICT[key] = ''
            elif key:
                msa_A_DICT[key] += line

    msa_B_DICT = {}
    with open(chainB_msa, 'r') as msafile_B:
        for line in msafile_B:
            line = line.strip()
            if line.startswith('>'):
                key = line
                msa_B_DICT[key] = ''
            elif key:
                msa_B_DICT[key] += line

    merged_dict = {}
    keys_A = list(msa_A_DICT.keys())
    keys_B = list(msa_B_DICT.keys())

    if len(keys_A) == len(keys_B):
        for i in range(len(keys_A)):
            key_A = keys_A[i]
            key_B = keys_B[i]
            key_B_splite = keys_B[i].split('>')[1]
            merged_key = f"{key_A}_____{key_B_splite}"
            print(merged_key)
            merged_value = msa_A_DICT[key_A] + msa_B_DICT[key_B]
            print(merged_value)
            merged_dict[merged_key] = merged_value

    with open(output_msa, 'w') as pairfile:
        for key, value in merged_dict.items():
            pairfile.write(f"{key}\n")
            pairfile.write(f"{value}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter and sort sequence pairs by score.')
    # input
    parser.add_argument('--input_score_csv', type=str, required=True,
                        help='Path to the input CSV file containing pairs and scores.')
    parser.add_argument('--input_msa_for_chain1', type=str, required=True,
                        help='Path to the first input FASTA file to extract sequences from (based on chain A).')
    parser.add_argument('--input_msa_for_chain2', type=str, required=True,
                        help='Path to the second input FASTA file to extract sequences from (based on chain B).')
    # output
    parser.add_argument('--output_pairs_csv', type=str, required=True,
                        help='Path to save the output filtered and sorted CSV file.')
    parser.add_argument('--output_msa_for_chain1', type=str, required=True,
                        help='Path to save the extracted sequences from the first input file.')
    parser.add_argument('--output_msa_for_chain2', type=str, required=True,
                        help='Path to save the extracted sequences from the second input file.')
    parser.add_argument('--output_complex_msa', type=str, required=True,
                        help='Path to the second input FASTA file to extract sequences from (based on chain B).')
    # others
    parser.add_argument('--index_1_start', type=int, default=1, help='Starting value for index_1. Default is 1.')
    parser.add_argument('--index_2_start', type=int, default=1, help='Starting value for index_2. Default is 1.')

    args = parser.parse_args()

    sorted_pairs = read_and_sort_pairs(args.input_score_csv)

    filtered_pairs = filter_unique_pairs(sorted_pairs, args.index_1_start, args.index_2_start)

    save_filtered_pairs_to_csv(filtered_pairs, args.output_pairs_csv)

    print(f"Filtered pairs saved to {args.output_pairs_csv}")

    sequence_ids_1 = read_sorted_pairs(args.output_pairs_csv, 0)
    sequence_ids_3 = read_sorted_pairs(args.output_pairs_csv, 2)

    sequences_1 = extract_sequences(args.input_msa_for_chain1, sequence_ids_1)
    sequences_3 = extract_sequences(args.input_msa_for_chain2, sequence_ids_3)

    save_sequences(args.output_msa_for_chain1, sequences_1)
    save_sequences(args.output_msa_for_chain2, sequences_3)

    complex_pair_msas_generation(args.output_msa_for_chain1, args.output_msa_for_chain2, args.output_complex_msa)
