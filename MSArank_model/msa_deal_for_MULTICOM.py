import csv
import argparse


def sort_scores(input_file, output_file):
    with open(input_file, mode='r') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        data = [row for row in reader]

    sorted_data = sorted(data, key=lambda x: float(x[2]), reverse=True)

    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(sorted_data)

    print(f"Sorted scores saved to {output_file}")
    return [row[1] for row in sorted_data]  # 返回按TargetSeqID排序的列表


def read_msa(msa_file):
    msa_dict = {}
    msa_order = []
    current_key = None
    current_sequence = []

    with open(msa_file, 'r') as infile:
        for line in infile:
            line = line.strip()
            if line.startswith('>'):
                if current_key is not None:
                    msa_dict[current_key] = ''.join(current_sequence)
                    msa_order.append(current_key)
                current_key = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)
        if current_key is not None:
            msa_dict[current_key] = ''.join(current_sequence)
            msa_order.append(current_key)

    return msa_dict, msa_order


def sort_msa(msa_dict, msa_order, sorted_target_ids):
    sorted_msa = []
    for target_id in sorted_target_ids:
        if target_id in msa_dict:
            sorted_msa.append(f'>{target_id}\n{msa_dict[target_id]}\n')
        else:
            print(f"Warning: {target_id} not found in MSA.")
    return sorted_msa


def save_sorted_msa(output_file, sorted_msa, new_fasta_header=None, new_fasta_sequence=None):
    with open(output_file, 'w') as outfile:
        if new_fasta_header and new_fasta_sequence:
            outfile.write(f'>{new_fasta_header}\n{new_fasta_sequence}\n')
        outfile.writelines(sorted_msa)
    print(f"Sorted MSA saved to {output_file}")


def read_fasta(fasta_file):
   with open(fasta_file, 'r') as infile:
        header = None
        sequence = []
        for line in infile:
            line = line.strip()
            if line.startswith('>'):
                header = line[1:]
            else:
                sequence.append(line)
        sequence = ''.join(sequence)
    return header, sequence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sort final_scores.csv and MSA file based on Score and TargetSeqID")
    parser.add_argument('--input_csv_file', type=str, required=True, help="Path to the input CSV file (final_scores.csv)")
    parser.add_argument('--output_csv_file', type=str, required=True, help="Path to save the sorted CSV file")
    parser.add_argument('--input_msa_file', type=str, required=True, help="Path to the input MSA file")
    parser.add_argument('--output_sorted_msa_file', type=str, required=True, help="Path to save the sorted MSA file")
    parser.add_argument('--input_fasta_file', type=str, required=True, help="Path to the new fasta file to be added to MSA")

    args = parser.parse_args()

    # Step 1: Sort final_scores.csv based on Score
    sorted_target_ids = sort_scores(args.input_csv_file, args.output_csv_file)

    # Step 2: Read the MSA file
    msa_dict, msa_order = read_msa(args.input_msa_file)

    # Step 3: Sort the MSA based on sorted TargetSeqID
    sorted_msa = sort_msa(msa_dict, msa_order, sorted_target_ids)

    # Step 4: Read the new fasta file to be added
    fasta_header, fasta_sequence = read_fasta(args.input_fasta_file)

    # Step 5: Save the sorted MSA to a new file and add the fasta sequence to the top
    save_sorted_msa(args.output_sorted_msa_file, sorted_msa, new_fasta_header=fasta_header, new_fasta_sequence=fasta_sequence)
