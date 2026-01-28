# Exposing the Unseen Layers of the Sun during Solar Disruptions of Oct-2024

- Dare to Dream
- Inspire Bold Leadership
- Embrace Diversity in Thought and Innovation
- Explore & Unveil the Depths of Interplanetary Knowledge

----------------------

This idea introduces an unconventional method for segmenting solar images using advanced techniques to isolate critical features like sunspots and solar flares. By effectively slicing images into meaningful segments, the approach enhances the analysis and interpretation of solar data, improving predictive models of solar activity.

The method not only deepens our understanding of solar phenomena but also holds promise for advancing space weather forecasting and the broader study of astrophysical processes.

<!--_Note: The code was run to generate Solar Slices on my personal MacBook Pro (Retina, Mid 2012)._-->

-----------------------

## Exposing the Unseen Layers of the Sun during Solar Disruptions October 2024

### GOES-16SUVI-2024-10-09-20:24:35

_source: https://www.swpc.noaa.gov/_

![alt text](image-3.png)

---------------
#### Solar Slices from GOES-16SUVI-2024-10-09-20:24:35

<!--_Note: The images below were sliced on my personal MacBook Pro (Retina, Mid 2012)._-->

![alt text](image-4.png)


### GOES-16SUVI-2024-10-11-02:04:36

_source: https://www.swpc.noaa.gov/_

![alt text](image.png)

---------------

#### Solar Slices from GOES-16SUVI-2024-10-11-02:04:36

<!--_Note: The images below were sliced on my personal MacBook Pro (Retina, Mid 2012)._-->

![alt text](image-1.png)

### GOES-16SUVI-2024-10-13-16:48:38

_source: https://www.swpc.noaa.gov/_

![alt text](image-5.png)

---------------

#### Solar Slices from GOES-16SUVI-2024-10-13-16:48:38

<!--_Note: The images below were sliced on my personal MacBook Pro (Retina, Mid 2012)._-->

![alt text](image-2.png)

---------------

_Additional Slices: https://github.com/ai-engineering-lab/Exposing-the-Unseen-Layers-of-the-Sun-during-Solar-Disruptions/tree/main/slices_

-----------------------

## Virtual environment (venv)

Create and use a project venv:

```bash
# Create
python3 -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

On Windows: `.venv\Scripts\activate`. The `.venv/` directory is listed in `.gitignore`.

-----------------------

## Solar slicing (Python): NMF, PCA, ICA + clustering

Unsupervised decomposition and clustering are implemented in `solar_slicing.py`. It produces separate “layer” images using:

- **PCA** – principal components over pixel vectors
- **NMF** – non-negative matrix factorization
- **ICA** – independent component analysis
- **K-means** – clustering on (intensity, x, y) for single-image mode

**Setup:** Activate the venv, then `pip install -r requirements.txt` (numpy, scikit-learn, Pillow).

**Run (default: stack of `image.png`, `image-3.png`, `image-5.png`):**
```bash
python3 solar_slicing.py --out output_slices
```

**Single image (decomposition + K-means layers):**
```bash
python3 solar_slicing.py --single image.png --out output_slices --components 3 --clusters 4
```

**Explicit image list:**
```bash
python3 solar_slicing.py --images image.png image-3.png image-5.png --out output_slices --components 3
```

Outputs are saved as PNGs in the given `--out` directory (e.g. `stack_pca_1.png`, `stack_nmf_2.png`, `kmeans_1.png`, …).
