## generate the cleaned data
python2 make_dataset.py \
        --ecb_path ../data/original/ECB+_LREC2014/ECB+ \
        --output_dir ../data/cleaned \
        --data_setup 2 \
        --selected_sentences_file ../data/original/ECB+_LREC2014/ECBplus_coreference_sentences.csv

## generate the pickle dumped corpus object
python3 build_features.py \
        --config_path build_features_config.json \
        --output_path ../data/pkl_data

## generate the event-pair samples
python3 generate_samples.py \
        --seed 2020 \
        --data_dir "../data/pkl_data" \
        --predicted_topics "../data/topics/predicted_topics"\
        --output_dir "../data/pairwise"