package ai.h2o.targetencoding;

import ai.h2o.targetencoding.TargetEncoderModel.DataLeakageHandlingStrategy;
import ai.h2o.targetencoding.TargetEncoderModel.TargetEncoderOutput;
import ai.h2o.targetencoding.TargetEncoderModel.TargetEncoderParameters;
import hex.ModelBuilder;
import hex.ModelCategory;
import water.DKV;
import water.Key;
import water.Scope;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;
import water.logging.Logger;
import water.logging.LoggerFactory;
import water.util.IcedHashMap;

import java.util.*;

import static ai.h2o.targetencoding.TargetEncoderHelper.*;
import static ai.h2o.targetencoding.TargetEncoderModel.NA_POSTFIX;

public class TargetEncoder extends ModelBuilder<TargetEncoderModel, TargetEncoderParameters, TargetEncoderOutput> {

  private static final Logger logger = LoggerFactory.getLogger(TargetEncoder.class);
  private TargetEncoderModel _targetEncoderModel;
  private String[][] _columnsToEncode;
  
  public TargetEncoder(TargetEncoderParameters parms) {
    super(parms);
    init(false);
  }

  public TargetEncoder(TargetEncoderParameters parms, Key<TargetEncoderModel> key) {
    super(parms, key);
    init(false);
  }

  public TargetEncoder(final boolean startupOnce) {
    super(new TargetEncoderParameters(), startupOnce);
  }

  @Override
  public void init(boolean expensive) {
    disableIgnoreConstColsFeature(expensive);
    ignoreUnusedColumns(expensive);
    super.init(expensive);
    assert _parms._nfolds == 0 : "nfolds usage forbidden in TargetEncoder";
    
    if (expensive) {
      if (_parms._data_leakage_handling == null) _parms._data_leakage_handling = DataLeakageHandlingStrategy.None;
      if (_parms._data_leakage_handling == DataLeakageHandlingStrategy.KFold && _parms._fold_column == null)
        error("_fold_column", "Fold column is required when using KFold leakage handling strategy.");

      final Frame train = train();
      
      _columnsToEncode = _parms._columns_to_encode;
      if (_columnsToEncode == null) { // detects columns that can be encoded
        final List<String> colsToIgnore = Arrays.asList(
                _parms._response_column,
                _parms._fold_column,
                _parms._weights_column,
                _parms._offset_column
        );
        final List<String[]> columnsToEncode = new ArrayList<>(train.numCols());
        for (int i = 0; i < train.numCols(); i++) {
          String colName = train.name(i);
          if (colsToIgnore.contains(colName)) continue;
          if (!train.vec(i).isCategorical()) {
            warn("_train", "Column `" + colName + "` is not categorical and will therefore be ignored by target encoder.");
            continue;
          }
          columnsToEncode.add(new String[] {colName});
        }
        _columnsToEncode = columnsToEncode.toArray(new String[0][]);
      } else { // validates column groups (which can be single columns)
        Set<String> validated = new HashSet<>();
        for (String[] colGroup: _columnsToEncode) {
          if (colGroup.length != new HashSet<>(Arrays.asList(colGroup)).size()) {
            error("_columns_to_encode", "Columns interaction "+Arrays.toString(colGroup)+" contains duplicate columns.");
          }
          for (String col: colGroup) {
            if (!validated.contains(col)) {
              if (!train.vec(col).isCategorical()) {
                error("_columns_to_encode", "Column "+col+" must first be converted into categorical to be used by target encoder.");
              }
              validated.add(col);
            }
          }
        }
      }
    }
  }

  private void disableIgnoreConstColsFeature(boolean expensive) {
    _parms._ignore_const_cols = false;
    if (expensive && logger.isInfoEnabled())
      logger.info("We don't want to ignore any columns during target encoding transformation " + 
              "therefore `_ignore_const_cols` parameter was set to `false`");
  }

  /**
   * autosets _ignored_columns when using _columns_to_encode param.
   * This ensures consistency when using the second param, 
   * otherwise the metadata saved in the model (_domains, _names...) can be different, 
   * and the score/predict result frame is also adapted differently.
   * @param expensive
   */
  private void ignoreUnusedColumns(boolean expensive) {
    if (!expensive || _parms._columns_to_encode == null || _parms.train() == null) return;
    Set<String> usedColumns = new HashSet<>(Arrays.asList(_parms.getNonPredictors()));
    for (String[] colGroup: _parms._columns_to_encode) usedColumns.addAll(Arrays.asList(colGroup));
    Set<String> unusedColumns = new HashSet<>(Arrays.asList(_parms.train()._names));
    unusedColumns.removeAll(usedColumns);
    _parms._ignored_columns = unusedColumns.toArray(new String[0]);
  }
  
  
  private class TargetEncoderDriver extends Driver {
    @Override
    public void computeImpl() {
      _targetEncoderModel = null;
      try {
        init(true);
        if (error_count() > 0)
          throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(TargetEncoder.this);

        TargetEncoderOutput emptyOutput = new TargetEncoderOutput(TargetEncoder.this);
        TargetEncoderModel model = new TargetEncoderModel(dest(), _parms, emptyOutput);
        _targetEncoderModel = model.delete_and_lock(_job); // and clear & write-lock it (smashing any prior)
        
        Frame workingFrame = new Frame(train());
        ColumnsMapping[] columnsToEncodeMapping = new ColumnsMapping[_columnsToEncode.length];
        for (int i=0; i < columnsToEncodeMapping.length; i++) {
          String[] colGroup = _columnsToEncode[i];
          String encodingCol = createFeatureInteraction(workingFrame, colGroup, _parms._encode_unseen_as_na_in_interactions);
          columnsToEncodeMapping[i] = new ColumnsMapping(colGroup, encodingCol);
        }
        
        String[] singleColumnsToEncode = Arrays.stream(columnsToEncodeMapping).map(ColumnsMapping::toSingle).toArray(String[]::new);
        IcedHashMap<String, Frame> _targetEncodingMap = prepareEncodingMap(workingFrame, singleColumnsToEncode);

        for (Map.Entry<String, Frame> entry : _targetEncodingMap.entrySet()) {
          Frame encodings = entry.getValue();
          Scope.untrack(encodings);
        }

        _targetEncoderModel._output = new TargetEncoderOutput(TargetEncoder.this, _targetEncodingMap,  columnsToEncodeMapping);
        _job.update(1);
      } catch (Exception e) {
        if (_targetEncoderModel != null) {
            Scope.track_generic(_targetEncoderModel);
        }
        throw e;
      } finally {
        if (_targetEncoderModel != null) {
          _targetEncoderModel.update(_job);
          _targetEncoderModel.unlock(_job);
        }
      }
    }
    
    //TODO We might want to introduce parameter that will change this behaviour. We can treat NA's as extra class.
    private Frame filterOutNAsFromTargetColumn(Frame data, int targetColumnIndex) {
      return filterOutNAsInColumn(data, targetColumnIndex);
    }


    private IcedHashMap<String, Frame> prepareEncodingMap(Frame fr, String[] columnsToEncode) {
      Frame workingFrame = null;
      try {
        int targetIdx = fr.find(_parms._response_column);
        int foldColIdx = _parms._fold_column == null ? -1 : fr.find(_parms._fold_column);
        
        workingFrame = filterOutNAsFromTargetColumn(fr, targetIdx);

        IcedHashMap<String, Frame> columnToEncodings = new IcedHashMap<>();

        for (String columnToEncode : columnsToEncode) { // TODO: parallelize
          int colIdx = workingFrame.find(columnToEncode);
          imputeCategoricalColumn(workingFrame, colIdx, columnToEncode + NA_POSTFIX);
          Frame encodings = buildEncodingsFrame(workingFrame, colIdx, targetIdx, foldColIdx, nclasses());

          Frame finalEncodings = applyLeakageStrategyToEncodings(
                  encodings, 
                  columnToEncode, 
                  _parms._data_leakage_handling, 
                  _parms._fold_column
          );
          encodings.delete();
          encodings = finalEncodings;
          if (encodings._key != null) DKV.remove(encodings._key);
          encodings._key = Key.make(_result.toString()+"_encodings_"+columnToEncode);
          DKV.put(encodings);
          columnToEncodings.put(columnToEncode, encodings);
        }
        
        return columnToEncodings;
      } finally {
        if (workingFrame != null) workingFrame.delete();
      }
    }

    private Frame applyLeakageStrategyToEncodings(Frame encodings, String columnToEncode, 
                                                  DataLeakageHandlingStrategy leakageHandlingStrategy, String foldColumn) {
      Frame groupedEncodings = null;
      int encodingsTEColIdx = encodings.find(columnToEncode);

      try {
        Scope.enter();
        switch (leakageHandlingStrategy) {
          case KFold:
            long[] foldValues = getUniqueColumnValues(encodings, encodings.find(foldColumn));
            for (long foldValue : foldValues) {
              Frame outOfFoldEncodings = getOutOfFoldEncodings(encodings, foldColumn, foldValue);
              Scope.track(outOfFoldEncodings);
              Frame tmpEncodings = register(groupEncodingsByCategory(outOfFoldEncodings, encodingsTEColIdx));
              Scope.track(tmpEncodings);
              addCon(tmpEncodings, foldColumn, foldValue); //groupEncodingsByCategory always removes the foldColumn, so we can reuse the same name immediately

              if (groupedEncodings == null) {
                groupedEncodings = tmpEncodings;
              } else {
                Frame newHoldoutEncodings = rBind(groupedEncodings, tmpEncodings);
                groupedEncodings.delete();
                groupedEncodings = newHoldoutEncodings;
              }
              Scope.track(groupedEncodings);
            }
            break;

          case LeaveOneOut:
          case None:
            groupedEncodings = groupEncodingsByCategory(encodings, encodingsTEColIdx, foldColumn != null);
            break;

          default:
            throw new IllegalStateException("null or unsupported leakageHandlingStrategy");
        }
        Scope.untrack(groupedEncodings);
      } finally {
        Scope.exit();
      }

      return groupedEncodings;
    }

    private Frame getOutOfFoldEncodings(Frame encodingsFrame, String foldColumn, long foldValue)  {
      int foldColumnIdx = encodingsFrame.find(foldColumn);
      return filterNotByValue(encodingsFrame, foldColumnIdx, foldValue);
    }

  }

  @Override
  protected void ignoreInvalidColumns(int npredictors, boolean expensive) {
    new FilterCols(npredictors){
      @Override
      protected boolean filter(Vec v) {
        return !v.isCategorical();
      }
    }.doIt(train(), "Removing non-categorical columns found in the list of encoded columns.", expensive);
  }

  /**
   * Never do traditional cross-validation for Target Encoder Model. The {@link TargetEncoderHelper} class handles
   * fold column on it's own.
   *
   * @return Always false
   */
  @Override
  public boolean nFoldCV() {
    return false;
  }

  @Override
  protected Driver trainModelImpl() {
    // We can use Model.Parameters to configure Target Encoder
    return new TargetEncoderDriver();
  }
  
  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[]{ ModelCategory.TargetEncoder};
  }

  @Override
  public boolean isSupervised() {
    return true;
  }

  @Override
  public BuilderVisibility builderVisibility() {
    return BuilderVisibility.Stable;
  }

  @Override
  public boolean haveMojo() {
    return true;
  }

}
