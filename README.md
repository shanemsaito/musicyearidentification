# Music Decade Identification
Deep learning model that can predict the musical decade of a song at 92.7% accuracy developed using PyTorch. Training data derived from the Spotify Web API.

## Dataset

This data set was created using Spotify's Web API. There are 700 songs total, with 100 each from the 1950s-2010s. Each song is analyzed using its average loudness in decibels (dBs) and 12 highly abstracted "timbre" values derived from Spotify's audio analysis features. 

These features are represented by the diagram below:

![](https://github.com/shanemsaito/musicyearidentification/blob/main/98b920868ca0ed105f739fa53c99adbd-2.png)

An explanation is provided by the [Spotify Web API](https://developer.spotify.com/documentation/web-api/reference/get-audio-analysis) documentation:

"The timbre feature is a vector that includes 12 unbounded values roughly centered around 0. Those values are high level abstractions of the spectral surface, ordered by degree of importance. For completeness however, the first dimension represents the average loudness of the segment; second emphasizes brightness; third is more closely correlated to the flatness of a sound; fourth to sounds with a stronger attack; etc...Timbre vectors are best used in comparison with each other."

To represent entire songs using these vectors, I took the average of all songs' timbre vectors so each song only has one respective timbre vector.

## Model

The baseline version of this model uses a basic neural network with two hidden linear layers each with a ReLU activation.

```python
class Model(nn.Module):
  def __init__(self, in_features = 14, h1 = 8, h2 = 9, out_features = 7):
    super().__init__()
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x
```

We train the model using 200 epochs which, for this baseline model, is an arbitrary choice based on graphing losses over epochs.

## Results

This model achieves 92.7% accuracy (127 / 137 testing songs). 

## Further Improvements

Because this is an elementary model with subpar data preprocessing, there a few benefits that may be implemented next:

1. Storing all timbre vectors of each song and padding shorter songs with 0s when necessary
2. Using temporal neural networks to identify changes over time within songs with the new sequential list of timbre vectors
3. Determining the optimal amount of epochs more rigorously by observing the models performance on the testing data with each epoch.


