data=("ag_news")
al_model=("RoBERTa")
sampling=("bemps")
n_train=(1 2 3 4 5)



for i in "${n_train[@]}"; do
    echo $i $data $al_model $sampling
    python trainALC.py --conf conf/${data}.json \
                        --output_dir output_exp/mollia_output/${data}_${al_model}_${sampling}/ \
                        --al_model ${al_model} \
                        --dataset ${data}\
                        --num_al 12 \
                        --num_epochs 4 \
                        --n_train ${i} \
                        --sampling ${sampling} \
                        --n_annote 5 \
                        --n 10 \
                        --temp 0.7 \
                        --prob 0.9
done