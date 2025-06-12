# Efficient Proxy Raytracer for Optical Systems using Implicit Neural Representations

**Abstract:**

Ray tracing is a widely used technique for modeling optical systems, involving 
sequential surface-by-surface computations which can be computationally 
intensive.
We propose Ray2Ray, a novel method that leverages implicit neural representations to 
model optical systems with greater efficiency, eliminating the need for 
surface-by-surface computations in a single pass end-to-end model.
Ray2Ray learns the mapping between rays emitted from a given source and their 
corresponding rays after passing through a given optical system in a physically 
accurate manner.

<div style="text-align: center;">
  <img src="media/repre.jpg" alt="Representative Image" width="500" >
</div>
[**Manuscript**](#)


# Getting Started

## Requirements


Install dependencies using pip or use following .yml file:

```bash
pip install -r requirements.txt
```


## Steps
**RayTracing** 


You can use our built-in ray tracer or integrate your own. Just ensure your output data matches our expected structure to continue using this repository.

To run our built-in ray tracer, use the following command. An example JSON file is provided in the lenses folder for reference.

```bash 
python raytracing_sys.py --lens_file  your_directory_to_lens.json --output_directory  save_rays_directory
```
For furthur explanation please use --help.

**Preparing Dataset** 

Use following command to prepare your dataset. 
```bash 
python dataset_maker.py --base_directory output_raytracing_dir --output_dir  save_train_test_dir
```
For furthur explanation please use --help.

**Train your INR optical system**

```bash 
python train.py --data_root_dir root_data_directory --data_folder  data_for_your_optical_system_dir
```
For additional options and command-line arguments for training, use --help.

**Test and explore your INR optical system**

```bash 
python test.py --data_root_dir root_data_directory --data_folder  data_for_your_optical_system_dir --model_path   INR_opticalsystem_path.pth
```
For additional options and command-line arguments for training, use --help.

# Support
This repository is actively being developed. I'm working to improve features and address issues. If you encounter any problems or have questions, feel free to contact shivansinaei@gmail.com or raise an issue.


