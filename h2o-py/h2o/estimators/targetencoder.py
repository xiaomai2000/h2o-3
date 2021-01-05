#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# This file is auto-generated by h2o-3/h2o-bindings/bin/gen_python.py
# Copyright 2016 H2O.ai;  Apache License Version 2.0 (see LICENSE for details)
#
from __future__ import absolute_import, division, print_function, unicode_literals

import h2o
import warnings
from h2o.exceptions import H2ODeprecationWarning
from h2o.utils.metaclass import deprecated_property
from h2o.utils.typechecks import U
from h2o.estimators.estimator_base import H2OEstimator
from h2o.exceptions import H2OValueError
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type, Enum, numeric


class H2OTargetEncoderEstimator(H2OEstimator):
    """
    TargetEncoder

    """

    algo = "targetencoder"
    param_names = {"model_id", "training_frame", "fold_column", "response_column", "ignored_columns",
                   "columns_to_encode", "keep_original_categorical_columns", "blending", "inflection_point",
                   "smoothing", "data_leakage_handling", "noise", "seed"}

    def __init__(self, **kwargs):
        super(H2OTargetEncoderEstimator, self).__init__()
        self._parms = {}
        for pname, pvalue in kwargs.items():
            if pname == 'model_id':
                self._id = pvalue
                self._parms["model_id"] = pvalue
            elif pname in self._deprecated_params_:
                setattr(self, pname, pvalue)  # property handles the redefinition
            elif pname in self.param_names:
                # Using setattr(...) will invoke type-checking of the arguments
                setattr(self, pname, pvalue)
            else:
                raise H2OValueError("Unknown parameter %s = %r" % (pname, pvalue))

    @property
    def training_frame(self):
        """
        Id of the training data frame.

        Type: ``H2OFrame``.

        :examples:

        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")
        >>> predictors = ["home.dest", "cabin", "embarked"]
        >>> response = "survived"
        >>> titanic["survived"] = titanic["survived"].asfactor()
        >>> fold_col = "kfold_column"
        >>> titanic[fold_col] = titanic.kfold_column(n_folds=5, seed=1234)
        >>> titanic_te = H2OTargetEncoderEstimator(inflection_point=35,
        ...                                        smoothing=25,
        ...                                        blending=True)
        >>> titanic_te.train(x=predictors,
        ...                  y=response,
        ...                  training_frame=titanic)
        >>> titanic_te
        """
        return self._parms.get("training_frame")

    @training_frame.setter
    def training_frame(self, training_frame):
        self._parms["training_frame"] = H2OFrame._validate(training_frame, 'training_frame')


    @property
    def fold_column(self):
        """
        Column with cross-validation fold index assignment per observation.

        Type: ``str``.

        :examples:

        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")
        >>> predictors = ["home.dest", "cabin", "embarked"]
        >>> response = "survived"
        >>> titanic["survived"] = titanic["survived"].asfactor()
        >>> fold_col = "kfold_column"
        >>> titanic[fold_col] = titanic.kfold_column(n_folds=5, seed=1234)
        >>> titanic_te = H2OTargetEncoderEstimator(inflection_point=35,
        ...                                        smoothing=25,
        ...                                        blending=True)
        >>> titanic_te.train(x=predictors,
        ...                  y=response,
        ...                  training_frame=titanic)
        >>> titanic_te
        """
        return self._parms.get("fold_column")

    @fold_column.setter
    def fold_column(self, fold_column):
        assert_is_type(fold_column, None, str)
        self._parms["fold_column"] = fold_column


    @property
    def response_column(self):
        """
        Response variable column.

        Type: ``str``.
        """
        return self._parms.get("response_column")

    @response_column.setter
    def response_column(self, response_column):
        assert_is_type(response_column, None, str)
        self._parms["response_column"] = response_column


    @property
    def ignored_columns(self):
        """
        Names of columns to ignore for training.

        Type: ``List[str]``.
        """
        return self._parms.get("ignored_columns")

    @ignored_columns.setter
    def ignored_columns(self, ignored_columns):
        assert_is_type(ignored_columns, None, [str])
        self._parms["ignored_columns"] = ignored_columns


    @property
    def columns_to_encode(self):
        """
        List of categorical columns or groups of categorical columns to encode. When groups of columns are specified,
        each group is encoded as a single column (interactions are created internally).

        Type: ``List[List[str]]``.
        """
        return self._parms.get("columns_to_encode")

    @columns_to_encode.setter
    def columns_to_encode(self, columns_to_encode):
        assert_is_type(columns_to_encode, None, [U(str, [str])])
        if columns_to_encode:  # standardize as a nested list
            columns_to_encode = [[g] if isinstance(g, str) else g for g in columns_to_encode]
        self._parms["columns_to_encode"] = columns_to_encode


    @property
    def keep_original_categorical_columns(self):
        """
        If true, the original non-encoded categorical features will remain in the result frame.

        Type: ``bool``  (default: ``True``).
        """
        return self._parms.get("keep_original_categorical_columns")

    @keep_original_categorical_columns.setter
    def keep_original_categorical_columns(self, keep_original_categorical_columns):
        assert_is_type(keep_original_categorical_columns, None, bool)
        self._parms["keep_original_categorical_columns"] = keep_original_categorical_columns


    @property
    def blending(self):
        """
        If true, enables blending of posterior probabilities (computed for a given categorical value) with prior
        probabilities (computed on the entire set). This allows to mitigate the effect of categorical values with small
        cardinality. The blending effect can be tuned using the `inflection_point` and `smoothing` parameters.

        Type: ``bool``  (default: ``False``).

        :examples:

        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")
        >>> predictors = ["home.dest", "cabin", "embarked"]
        >>> response = "survived"
        >>> titanic["survived"] = titanic["survived"].asfactor()
        >>> fold_col = "kfold_column"
        >>> titanic[fold_col] = titanic.kfold_column(n_folds=5, seed=1234)
        >>> titanic_te = H2OTargetEncoderEstimator(inflection_point=35,
        ...                                        smoothing=25,
        ...                                        blending=True)
        >>> titanic_te.train(x=predictors,
        ...                  y=response,
        ...                  training_frame=titanic)
        >>> titanic_te
        """
        return self._parms.get("blending")

    @blending.setter
    def blending(self, blending):
        assert_is_type(blending, None, bool)
        self._parms["blending"] = blending


    @property
    def inflection_point(self):
        """
        Inflection point of the sigmoid used to blend probabilities (see `blending` parameter). For a given categorical
        value, if it appears less that `inflection_point` in a data sample, then the influence of the posterior
        probability will be smaller than the prior.

        Type: ``float``  (default: ``10``).

        :examples:

        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")
        >>> predictors = ["home.dest", "cabin", "embarked"]
        >>> response = "survived"
        >>> titanic["survived"] = titanic["survived"].asfactor()
        >>> fold_col = "kfold_column"
        >>> titanic[fold_col] = titanic.kfold_column(n_folds=5, seed=1234)
        >>> titanic_te = H2OTargetEncoderEstimator(inflection_point=35,
        ...                                        smoothing=25,
        ...                                        blending=True)
        >>> titanic_te.train(x=predictors,
        ...                  y=response,
        ...                  training_frame=titanic)
        >>> titanic_te
        """
        return self._parms.get("inflection_point")

    @inflection_point.setter
    def inflection_point(self, inflection_point):
        assert_is_type(inflection_point, None, numeric)
        self._parms["inflection_point"] = inflection_point


    @property
    def smoothing(self):
        """
        Smoothing factor corresponds to the inverse of the slope at the inflection point on the sigmoid used to blend
        probabilities (see `blending` parameter). If smoothing tends towards 0, then the sigmoid used for blending turns
        into a Heaviside step function.

        Type: ``float``  (default: ``20``).

        :examples:

        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")
        >>> predictors = ["home.dest", "cabin", "embarked"]
        >>> response = "survived"
        >>> titanic["survived"] = titanic["survived"].asfactor()
        >>> fold_col = "kfold_column"
        >>> titanic[fold_col] = titanic.kfold_column(n_folds=5, seed=1234)
        >>> titanic_te = H2OTargetEncoderEstimator(inflection_point=35,
        ...                                        smoothing=25,
        ...                                        blending=True)
        >>> titanic_te.train(x=predictors,
        ...                  y=response,
        ...                  training_frame=titanic)
        >>> titanic_te
        """
        return self._parms.get("smoothing")

    @smoothing.setter
    def smoothing(self, smoothing):
        assert_is_type(smoothing, None, numeric)
        self._parms["smoothing"] = smoothing


    @property
    def data_leakage_handling(self):
        """
        Data leakage handling strategy used to generate the encoding. Supported options are: 1) "none" (default) - no
        holdout, using the entire training frame. 2) "leave_one_out" - current row's response value is subtracted from
        the per-level frequencies pre-calculated on the entire training frame. 3) "k_fold" - encodings for a fold are
        generated based on out-of-fold data.

        One of: ``"leave_one_out"``, ``"k_fold"``, ``"none"``  (default: ``"none"``).

        :examples:

        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")
        >>> predictors = ["home.dest", "cabin", "embarked"]
        >>> response = "survived"
        >>> titanic["survived"] = titanic["survived"].asfactor()
        >>> fold_col = "kfold_column"
        >>> titanic[fold_col] = titanic.kfold_column(n_folds=5, seed=1234)
        >>> titanic_te = H2OTargetEncoderEstimator(inflection_point=35,
        ...                                        smoothing=25,
        ...                                        data_leakage_handling="k_fold",
        ...                                        blending=True)
        >>> titanic_te.train(x=predictors,
        ...                  y=response,
        ...                  training_frame=titanic)
        >>> titanic_te
        """
        return self._parms.get("data_leakage_handling")

    @data_leakage_handling.setter
    def data_leakage_handling(self, data_leakage_handling):
        assert_is_type(data_leakage_handling, None, Enum("leave_one_out", "k_fold", "none"))
        self._parms["data_leakage_handling"] = data_leakage_handling


    @property
    def noise(self):
        """
        The amount of noise to add to the encoded column. Use 0 to disable noise, and -1 (=AUTO) to let the algorithm
        determine a reasonable amount of noise.

        Type: ``float``  (default: ``0.01``).
        """
        return self._parms.get("noise")

    @noise.setter
    def noise(self, noise):
        assert_is_type(noise, None, numeric)
        self._parms["noise"] = noise


    @property
    def seed(self):
        """
        Seed used to generate the noise. By default, the seed is chosen randomly.

        Type: ``int``  (default: ``-1``).
        """
        return self._parms.get("seed")

    @seed.setter
    def seed(self, seed):
        assert_is_type(seed, None, int)
        self._parms["seed"] = seed


    _deprecated_params_ = ['k', 'f', 'noise_level']
    k = deprecated_property('k', inflection_point)
    f = deprecated_property('f', smoothing)
    noise_level = deprecated_property('noise_level', noise)

    def transform(self, frame, blending=None, inflection_point=None, smoothing=None, noise=None, as_training=False, **kwargs):
        """
        Apply transformation to `te_columns` based on the encoding maps generated during `train()` method call.

        :param H2OFrame frame: the frame on which to apply the target encoding transformations.
        :param boolean blending: If provided, this overrides the `blending` parameter on the model.
        :param float inflection_point: If provided, this overrides the `inflection_point` parameter on the model.
        :param float smoothing: If provided, this overrides the `smoothing` parameter on the model.
        :param float noise: If provided, this overrides the amount of random noise added to the target encoding defined on the model, this helps prevent overfitting.
        :param boolean as_training: Must be set to True when encoding the training frame. Defaults to False.

        :example:
        >>> titanic = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")
        >>> predictors = ["home.dest", "cabin", "embarked"]
        >>> response = "survived"
        >>> titanic[response] = titanic[response].asfactor()
        >>> fold_col = "kfold_column"
        >>> titanic[fold_col] = titanic.kfold_column(n_folds=5, seed=1234)
        >>> titanic_te = H2OTargetEncoderEstimator(data_leakage_handling="leave_one_out",
        ...                                        inflection_point=35,
        ...                                        smoothing=25,
        ...                                        blending=True,
        ...                                        seed=1234)
        >>> titanic_te.train(x=predictors,
        ...                  y=response,
        ...                  training_frame=titanic)
        >>> transformed = titanic_te.transform(frame=titanic)
        """
        for k in kwargs:
            if k in ['seed', 'data_leakage_handling']:
                warnings.warn("`%s` is deprecated in `transform` method and will be ignored. "
                              "Instead, please ensure that it was set before training on the H2OTargetEncoderEstimator model." % k, H2ODeprecationWarning)
            else:
                raise TypeError("transform() got an unexpected keyword argument '%s'" % k)

        if 'data_leakage_handling' in kwargs:
            dlh = kwargs['data_leakage_handling']
            assert_is_type(dlh, None, Enum("leave_one_out", "k_fold", "none"))
            if dlh is not None and dlh.lower() != "none":
                warnings.warn("Deprecated `data_leakage_handling=%s` is replaced by `as_training=True`. "
                              "Please update your code." % dlh, H2ODeprecationWarning)
                as_training = True

        params = dict(
            model=self.model_id,
            frame=frame.key,
            blending=blending if blending is not None else self.blending,  # always need to provide blending here as we can't represent unset value 
            inflection_point=inflection_point,
            smoothing=smoothing,
            noise=noise,
            as_training=as_training,
        )

        output = h2o.api("GET /3/TargetEncoderTransform", data=params)
        return h2o.get_frame(output["name"])
