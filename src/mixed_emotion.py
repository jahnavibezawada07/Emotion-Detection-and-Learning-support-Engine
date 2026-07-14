def get_mixed_emotions(scores, threshold=0.15):
    """
    Returns all emotions whose confidence is greater than or
    equal to the threshold.

    Parameters
    ----------
    scores : dict
        Example:
        {
            "Bored":0.02,
            "Confident":0.08,
            "Confused":0.32,
            "Curious":0.41,
            "Frustrated":0.17
        }

    threshold : float
        Minimum confidence to be considered a mixed emotion.
    """

    # Sort highest confidence first
    sorted_scores = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    primary_emotion = sorted_scores[0]

    mixed = [primary_emotion]

    for emotion, score in sorted_scores[1:]:

        if score >= threshold:

            mixed.append((emotion, score))

    return mixed