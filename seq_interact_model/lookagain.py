import argparse


def main():
    parser = argparse.ArgumentParser('Script for filt the a3m for ESM embeding vector')

    # input output option
    parser.add_argument('-input_a3m', type=str, default='/mydata/minghuah/MULTICOM/seq_interact_model/examples/H1106_B_uniref30_bfd.a3m',
                        help='input your filtef a3m')
    parser.add_argument('-output_a3m', type=str, default='/mydata/minghuah/MULTICOM/seq_interact_model/examples/H1106_B_uniref30_bfd_filted.a3m',
                        help='output the filted a3m')

    args = parser.parse_args()
    check(args.input_a3m, args.output_a3m)


def check(inputa3m, outputa3m):
    # 打开FASTA文件并读取内容
    with open(inputa3m, "r") as fasta_file:
        lines = fasta_file.readlines()

    # 创建一个列表来存储要保留的行
    filtered_lines = []

    # 创建一个字典来存储已经出现的标识行及其行号
    identifier_dict = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(">"):
            identifier = line.strip()  # 去除额外的空白
            if identifier not in identifier_dict:
                # 如果标识行没有出现过，保留当前行及下一行，并记录行号
                filtered_lines.append(line)
                i += 1
                filtered_lines.append(lines[i])
                identifier_dict[identifier] = i
            else:
                # 如果标识行已经出现过，打印重复的行号
                print("Duplicate identifier found at lines:", identifier_dict[identifier], i)
        i += 1

    # 创建一个新的文件，写入保留的行
    with open(outputa3m, "w") as filtered_fasta_file:
        for line in filtered_lines:
            filtered_fasta_file.write(line)


if __name__ == "__main__":
    main()
