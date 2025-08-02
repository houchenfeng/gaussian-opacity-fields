
# cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s DTU_mask/scan{scene} -m {output_dir}/scan{scene} -r {factor} --use_decoupled_appearance --lambda_distortion 1000"
CUDA_VISIBLE_DEVICES=0 python3 train.py -s /home/lt/2024/data/dtu_datasets/dtu_robust/scan24/data_0.3 \
                                        -m output/dtu_robust/scan24/data_0.3 -r 2 \
                                        --use_decoupled_appearance --lambda_distortion 1000
    
# marching tetrahedra with binary search
# cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python extract_mesh.py -m {output_dir}/scan{scene} --iteration 30000"
CUDA_VISIBLE_DEVICES=0 python3 extract_mesh_tsdf.py -m output/dtu_robust/scan24/data_0.3 --iteration 30000

# # tsdf fusion
# cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python extract_mesh_tsdf.py -m {output_dir}/scan{scene} --iteration 30000"


# # evaluate
# cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python evaluate_dtu_mesh.py -m {output_dir}/scan{scene} --iteration 30000"
