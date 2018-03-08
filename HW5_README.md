

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and inspect the data files
trial_file = "/Users/frimpter/Documents/data_science/ru_datascience_bootcamp/Homework5 Matplotlib/clinicaltrial_data.csv"
mice_file = "/Users/frimpter/Documents/data_science/ru_datascience_bootcamp/Homework5 Matplotlib/mouse_drug_data.csv"

trial_df = pd.DataFrame(pd.read_csv(trial_file))
mice_df = pd.DataFrame(pd.read_csv(mice_file))

#trial_df.head()
#mice_df.head()

# Set indices to Mouse ID and merge files
trial_df = trial_df.set_index("Mouse ID")
mice_df = mice_df.set_index("Mouse ID")

df = pd.merge(trial_df, mice_df, left_index=True, right_index=True)
df = df.reset_index()
df = df.rename(columns={"Timepoint":"Timepoint (days)"}) # Prepare variable name as axis label
#df.head()
```


```python
# TUMOR VOLUME: Group dataset by Drug and Timepoint to add SE values for each data point

tumor_vol = pd.DataFrame(round(df.groupby(["Drug", "Timepoint (days)"]).mean(), 2))
del tumor_vol["Metastatic Sites"]

# Add standard errors
se = tumor_vol["Tumor Volume (mm3)"].sem()

# Turn drugs into columns for graphing
tumor_vol = tumor_vol.unstack(level="Drug")

tumor_vol.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">Tumor Volume (mm3)</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint (days)</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>45.00</td>
      <td>45.00</td>
      <td>45.00</td>
      <td>45.00</td>
      <td>45.00</td>
      <td>45.00</td>
      <td>45.00</td>
      <td>45.00</td>
      <td>45.00</td>
      <td>45.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>44.27</td>
      <td>46.50</td>
      <td>47.06</td>
      <td>47.39</td>
      <td>46.80</td>
      <td>47.13</td>
      <td>47.25</td>
      <td>43.94</td>
      <td>47.53</td>
      <td>46.85</td>
    </tr>
    <tr>
      <th>10</th>
      <td>43.08</td>
      <td>48.29</td>
      <td>49.40</td>
      <td>49.58</td>
      <td>48.69</td>
      <td>49.42</td>
      <td>49.10</td>
      <td>42.53</td>
      <td>49.46</td>
      <td>48.69</td>
    </tr>
    <tr>
      <th>15</th>
      <td>42.06</td>
      <td>50.09</td>
      <td>51.30</td>
      <td>52.40</td>
      <td>50.93</td>
      <td>51.36</td>
      <td>51.07</td>
      <td>41.50</td>
      <td>51.53</td>
      <td>50.78</td>
    </tr>
    <tr>
      <th>20</th>
      <td>40.72</td>
      <td>52.16</td>
      <td>53.20</td>
      <td>54.92</td>
      <td>53.64</td>
      <td>54.36</td>
      <td>53.35</td>
      <td>40.24</td>
      <td>54.07</td>
      <td>53.17</td>
    </tr>
  </tbody>
</table>
</div>




```python
# GRAPH 1 - Tumor Volume by Drug

sns.set_palette(sns.color_palette("cubehelix", 10))

tumor_vol["Tumor Volume (mm3)"].plot(kind="line", ls="--", lw=2, ms=7, yerr=se, marker="o", grid=True, title="Tumor Response to Treatment", figsize=(10,10))

plt.ylabel("Tumor Volume (mm3)")
plt.ylim(30,80)

plt.show()
```


![png](output_2_0.png)



```python
# METASTATIC SITES: Group dataset by Drug and Timepoint to add SE values for mean number of sites

metsites = pd.DataFrame(round(df.groupby(["Drug", "Timepoint (days)"]).mean(), 2))
del metsites["Tumor Volume (mm3)"]

# Add standard errors
se = metsites["Metastatic Sites"].sem()

# Turn drugs into columns by timepoint for graphing
metsites = metsites.unstack(level="Drug")

metsites.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">Metastatic Sites</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint (days)</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.16</td>
      <td>0.38</td>
      <td>0.28</td>
      <td>0.30</td>
      <td>0.26</td>
      <td>0.38</td>
      <td>0.32</td>
      <td>0.12</td>
      <td>0.24</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.32</td>
      <td>0.60</td>
      <td>0.67</td>
      <td>0.59</td>
      <td>0.52</td>
      <td>0.83</td>
      <td>0.57</td>
      <td>0.25</td>
      <td>0.48</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.38</td>
      <td>0.79</td>
      <td>0.90</td>
      <td>0.84</td>
      <td>0.86</td>
      <td>1.25</td>
      <td>0.76</td>
      <td>0.33</td>
      <td>0.78</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.65</td>
      <td>1.11</td>
      <td>1.05</td>
      <td>1.21</td>
      <td>1.15</td>
      <td>1.53</td>
      <td>1.00</td>
      <td>0.35</td>
      <td>0.95</td>
      <td>1.29</td>
    </tr>
  </tbody>
</table>
</div>




```python
# GRAPH 2 - Mean Number of Metastatic Sites by Drug

metsites["Metastatic Sites"].plot(kind="line", ls="--", lw=2, ms=7, yerr=se, marker="o", grid=True, title="Metastatic Sites by Treatment", figsize=(10,10))

plt.ylabel("Number of Metastatic Sites, mean (SE)")
plt.ylim(0,5)

plt.show()
```


![png](output_4_0.png)



```python
# SURVIVAL RATES: Mouse survival at end of treatment period

survival = pd.DataFrame(round(df.groupby(["Drug", "Timepoint (days)"]).count(), 2))
del survival["Tumor Volume (mm3)"]
del survival["Metastatic Sites"]

survival = survival.unstack(level="Drug")

#survival

# Convert survival rates to percentages

survival_pct = round(pd.DataFrame(survival.iloc[:,:]/survival.iloc[0,:]*100),1)
survival_pct
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">Mouse ID</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint (days)</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>100.0</td>
      <td>84.0</td>
      <td>100.0</td>
      <td>92.0</td>
      <td>92.0</td>
      <td>96.0</td>
      <td>96.2</td>
      <td>100.0</td>
      <td>96.2</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>100.0</td>
      <td>80.0</td>
      <td>84.0</td>
      <td>88.0</td>
      <td>84.0</td>
      <td>96.0</td>
      <td>88.5</td>
      <td>96.0</td>
      <td>88.5</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>96.0</td>
      <td>76.0</td>
      <td>84.0</td>
      <td>76.0</td>
      <td>84.0</td>
      <td>80.0</td>
      <td>65.4</td>
      <td>96.0</td>
      <td>88.5</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>92.0</td>
      <td>72.0</td>
      <td>80.0</td>
      <td>76.0</td>
      <td>80.0</td>
      <td>76.0</td>
      <td>65.4</td>
      <td>92.0</td>
      <td>80.8</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>88.0</td>
      <td>72.0</td>
      <td>72.0</td>
      <td>76.0</td>
      <td>72.0</td>
      <td>68.0</td>
      <td>53.8</td>
      <td>92.0</td>
      <td>73.1</td>
      <td>64.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>88.0</td>
      <td>64.0</td>
      <td>68.0</td>
      <td>72.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>50.0</td>
      <td>92.0</td>
      <td>69.2</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>88.0</td>
      <td>56.0</td>
      <td>48.0</td>
      <td>68.0</td>
      <td>60.0</td>
      <td>56.0</td>
      <td>38.5</td>
      <td>84.0</td>
      <td>61.5</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>84.0</td>
      <td>56.0</td>
      <td>40.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>48.0</td>
      <td>34.6</td>
      <td>80.0</td>
      <td>46.2</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>84.0</td>
      <td>52.0</td>
      <td>36.0</td>
      <td>44.0</td>
      <td>52.0</td>
      <td>44.0</td>
      <td>26.9</td>
      <td>80.0</td>
      <td>42.3</td>
      <td>56.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Graph 3 - Survival of mice by treatment over study period

sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("cubehelix", 10))

survival_pct["Mouse ID"].plot(kind="line", ls="--", lw=2, ms=7, yerr=se, marker="o", grid=True, title="Survival Rate by Treatment (%)", figsize=(10,10))

plt.ylabel("Survival (%)")
plt.ylim(20,110)

plt.show()
```


![png](output_6_0.png)



```python
# TOTAL CHANGE IN TUMOR SIZE

# Use tumor_vol dataframe to calculate total percentage change in tumor size
tumor_change = pd.DataFrame(round(((tumor_vol.iloc[9,:]) - (tumor_vol.iloc[0,:])) / (tumor_vol.iloc[0,:])*100, 1))
tumor_change = tumor_change.sort_values(0)
tumor_change = tumor_change.unstack(level="Drug")

tumor_change
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">0</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Tumor Volume (mm3)</th>
      <td>-19.5</td>
      <td>42.5</td>
      <td>46.1</td>
      <td>57.0</td>
      <td>53.9</td>
      <td>51.3</td>
      <td>47.2</td>
      <td>-22.3</td>
      <td>52.1</td>
      <td>46.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Graph 4 - Total Change in Tumor Size over Study Period

plt.figure(figsize=(10,10))
sns.barplot(data=tumor_change[0], palette="Blues", order=["Ramicane", "Capomulin", "Ceftamin", "Infubinol", "Zoniferol", "Propriva", "Placebo", "Stelasyn", "Naftisol", "Ketapril"])

plt.title("Total Change in Tumor Volume by Treatment")

plt.xlabel("Drug")
plt.xlim(-1,10)
plt.ylabel("Tumor Volume Change from Baseline (%)")
plt.ylim(-30,60,10)

plt.hlines(0,-1,10)

plt.show()
```


![png](output_8_0.png)



```python
print("Three Trends from Pymaceuticals Analysis: ")
print("\n1. Only two drugs, Ramicane and Capomulin, reduced overall tumor size over the course of treatment.")
print("\n2. Ramicane and Capomulin also had the fewest metastatic sites at every timepoint in the study.")
print("\n3. Ramicane and Capomulin had the most favorable survival rates over all timepoints.")
```

    Three Trends from Pymaceuticals Analysis: 
    
    1. Only two drugs, Ramicane and Capomulin, reduced overall tumor size over the course of treatment.
    
    2. Ramicane and Capomulin also had the fewest metastatic sites at every timepoint in the study.
    
    3. Ramicane and Capomulin had the most favorable survival rates over all timepoints.

