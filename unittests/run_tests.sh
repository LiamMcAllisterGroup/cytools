# export OMP_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export VECLIB_MAXIMUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
python3 ./test_glsm.py
python3 ./test_kahler_positivity.py
python3 ./test_compare_poly_to_Sage.py
python3 ./test_compare_cy_to_Sage.py
