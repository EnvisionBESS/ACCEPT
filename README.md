# ACCEPT

Accurate modelling of lithium-ion battery degradation is a complex problem, dependent on multiple internal mechanisms that can be affected by a multitude of external conditions. Much research has been conducted into how machine learning can accurately forecast this. However, most of these approaches require large amounts of data, and this restricted form of supervision consistently show poor performance when the degradation is particularly high. Modelling degradation with well-understood physical properties remains an option, however small variances in physical parameters and uncertainty in these phenomena make it difficult to extrapolate degradation curves generated purely through physical models to real world operational data. To this end, we propose a new model - ACCEPT. Our model learns to map the underlying physical degradation parameters with observable operational quantities. This allows it to match a measured degradation curve with the best corresponding simulated scenario. By doing this, a potentially infinite number of physically generated curves, accounting for any scenario, can be matched to operational battery cell data. Due to the ability for cell chemistry to be entered as a category into the model, it can generalise to all main cell chemistries. Furthermore, due to the similarity of degradation paths between battery cells with the same chemistry, this model transfers non-trivially to most downstream tasks, allowing for zero-shot inference of future capacity degradation.

Through matching operational data to very similar degradation curves generated using known phyisical properties of abtteries, accept is able to quantify the degradation modes occuring in the battery from purely current, temperature and voltage readings, a feat previously not achieved by machine learning techniques to model degradation

## Approach 

![Pre-Training Technique](pictures/pre_training.png)

![TFT](pictures/tft.png)
## Data

So far ACCEPT has been trained on the following datasets, and will therefore yield good results for cells of similar size/chemistry:
* Severson 