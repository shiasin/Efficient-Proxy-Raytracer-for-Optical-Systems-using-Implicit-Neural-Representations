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

[**Manuscript**](#)


# Getting Started

## Requirements


Install dependencies using pip or use following .yml file:

```bash
pip install -r requirements.txt
```
or 
```bash
conda env create -f impenv.yml
```

## Steps
**RayTracing** 


You can use our built-in ray tracer or integrate your own. Just ensure your output data matches our expected structure to continue using this repository.

To run our built-in ray tracer, use the following command. An example JSON file is provided in the lenses folder for reference.

```bash 
python raytracing_sys --lens_file  your_directory_to_lens.json --output_directory  save_rays_directory
```
For furthur explanation please use --help.




