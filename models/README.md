# Models

The following models are built:
- `Popularity`: a naive recommender which recomemend the most rated movies (number of ratings given by users)
- `ContentBased`: recommend movies to users based on the most similar movies to the ones they have watched
- `UserBasedCF`: an user-based collaborative filtering model which recommend to users movies watched by others users similar to them
- `ItemBasedCF`: an item-based collaborative filtering model which recommend to users movies which are interacted in a similar way to the one they have given positive feedbacks
- `HybridExplicitFeedback`: a hybrid recommender which combines other explicit feedback recommenders to make recommendation
- `HybridImplicitFeedback`: a hybrid recommender of collaborative filtering and content-based recommenders. This recommender does not predict ratings. The main engine is [LightFM](https://making.lyst.com/lightfm/docs/lightfm.html)
