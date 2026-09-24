"""Species classification with spatial CV and Olofsson et al. (2014) area estimation."""


def train_species_classifier(X, y, groups, n_folds=5, seed=0):
    raise NotImplementedError("M1")


def olofsson_area(map_labels, ref_map, ref_true, n_classes, pixel_area_ha):
    raise NotImplementedError("M1")
