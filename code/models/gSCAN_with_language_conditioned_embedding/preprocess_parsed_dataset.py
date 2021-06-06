import json

# first parse the dataset following https://github.com/LauraRuis/multimodal_seq2seq_gSCAN/tree/master/read_gscan
# then runs this script

d = json.load(open('parsed_dataset/parsed_dataset.txt', 'r'))
for split, data in d.items():
    with open('parsed_dataset/' + split + '.json', 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')
