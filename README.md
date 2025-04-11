      [Image]
                ↓
      [Objective Features]
                ↓
      [Symbolic Features (Tier 1)]
                ↓
      [Compositional Description (Tier 2)]
                ↓
      [Vibe Atom Sentiment (Tier 0)]
                ↓
      [Mapped into Cultural Aesthetic (Tier 3)]
                ↓
      [Grouped under Archetypal Theme (Tier 4)]


Bidirectional -> mapping back to any other search/recommendation system: because each photo is also associated with a for example creator, we could allow for any tier to be mapped back to a specific image (or set of images) and then utilize that to identify certain creators/posts that fit this vibe.

Base Model

Comprehensive dictionary of baseline tier 0 vocabulary.

Create tier 1-4 derivatives and run Pintrest search -> create pictorial database. (100 images per query)

Extract OpenCV/PIL based factors for each image

Create objective based “vectors” for each image; perform clustering to see if images of the same baseline or derivative tiers cluster together. 

Embed all images using CLIP; overlay in UMAP/ISNE for interpretability

Overlay clustering on basis of different tiers -> see if derivative tiers map back to baseline tiers.

Isolate objective factors to see if specific objective factors or interactions of these factors lead to clustering for tiers.

The benefit of utilizing Pintrest searches is that they are rather inherently grouped by user-determined vibe.

The “vibe” factor is quite obvious.

Generalization

Utilizing OpenCV, CLIP/other pre-captioned databases to test model generalizability across fuzzier, less obvious photos. COCO, Flickr, OpenImages, FiveK

Generate Tier 2 phrases based on COCO

See how well Pintrest model generalizes

Running a Tiktok pipeline to allow for generalization across frames -> good self-captioning through the use of context by video description + hashtags.

Bidirectional Testing (?)

Demonstrating that a photo can be mapped from one tier to another?

3.1 
Classify vibes from symbolic and compositional input.

Generate Tier 2 descriptions.

Predict Tier 0/3/4 using intepretable classifiers or GPT

3.2 

Tier 2 text -> image retrieval via CLIP

Symbolic combo search: Tier 1 feature string -> vibe image set

Tier 3: predicted Tier 1/Tier 0 distribution

Mirroring With Language Sentiment (?)

Use text sentiment classification models (BERT/GPT on Tier 2)

Test cross modal sentiment embedding proxy (image <-> text)

Evaluate alignment between visual sentiment and textual tone

Vibe Drift Analysis

Take raw vs processed -> see shift -> continued in 6

Objective Factor Filter Overlay -> Analyze Changes in Vibe Analysis Based on This

Could take real life examples of reporting and analyze how differences in how a photo is taken, or how it’s post-processed might influence how users perceive a photo.
