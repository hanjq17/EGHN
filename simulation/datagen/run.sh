#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 5 --n_stick 0 --n_hinge 0 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 1 --n_stick 2 --n_hinge 0 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 2 --n_stick 0 --n_hinge 1 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 10 --n_stick 0 --n_hinge 0 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 4 --n_stick 3 --n_hinge 0 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 2 --n_stick 4 --n_hinge 0 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 0 --n_stick 5 --n_hinge 0 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 7 --n_stick 0 --n_hinge 1 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 4 --n_stick 0 --n_hinge 2 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 1 --n_stick 0 --n_hinge 3 --n_workers 50
#
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 20 --n_stick 0 --n_hinge 0 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 10 --n_stick 5 --n_hinge 0 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 8 --n_stick 6 --n_hinge 0 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 4 --n_stick 8 --n_hinge 0 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 0 --n_stick 10 --n_hinge 0 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 14 --n_stick 0 --n_hinge 2 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 8 --n_stick 0 --n_hinge 4 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 2 --n_stick 0 --n_hinge 6 --n_workers 50
#
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 3 --n_stick 2 --n_hinge 1 --n_workers 50
#
#python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 5 --n_stick 3 --n_hinge 3 --n_workers 50
#
#
#

# Small
python -u generate_dataset_complex.py --num-train 2000 --seed 43 --n_complex 5 --average_complex_size 3  --n_workers 50

python -u generate_dataset_complex.py --num-train 2000 --seed 43 --n_complex 8 --average_complex_size 5  --n_workers 50

python -u generate_dataset_complex.py --num-train 2000 --seed 43 --n_complex 10 --average_complex_size 10  --n_workers 50

# Median
python -u generate_dataset_complex.py --num-train 2000 --seed 43 --n_complex 5 --average_complex_size 10  --n_workers 50

python -u generate_dataset_complex.py --num-train 2000 --seed 43 --n_complex 5 --average_complex_size 10  --n_workers 50

# V100 96
# A100 192
