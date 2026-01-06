import 'package:tflite_flutter/tflite_flutter.dart';
// import 'dart:math';

class HarModelInfo {
  final String backend;
  final List<int> inputShape;
  final List<int> outputShape;
  final String inputDtype;
  final String outputDtype;
  final bool dryRunOk;
  final String? error;

  const HarModelInfo({
    required this.backend,
    required this.inputShape,
    required this.outputShape,
    required this.inputDtype,
    required this.outputDtype,
    required this.dryRunOk,
    this.error,
  });
}

class HARService {
  static Interpreter? _interpreter;

  static int _requiredWindow = 128;
  static int _numFeatures = 3;
  static int _numClasses = 11;

  static const List<String> _activityLabels = [
    'Standing',
    'Lying down on left',
    'Lying down right',
    'Lying down back',
    'Lying down on stomach',
    'Normal walking',
    'Ascending stairs',
    'Descending stairs',
    'Shuffle walking',
    'Running',
    'Miscellaneous movements'
  ];

  static List<List<double>> _sampleBuffer = [];

  static HarModelInfo? _modelInfo;
  static HarModelInfo? get modelInfo => _modelInfo;

  static Future<bool> initialize() async {
    try {
      _interpreter?.close();
      _interpreter = await Interpreter.fromAsset('har_model.tflite');
      final inT = _interpreter!.getInputTensor(0);
      final outT = _interpreter!.getOutputTensor(0);

      final inShape = List<int>.from(inT.shape);
      final outShape = List<int>.from(outT.shape);
      final inType = inT.type.toString().replaceFirst('TensorType.', '');
      final outType = outT.type.toString().replaceFirst('TensorType.', '');

      if (inShape.length >= 3) {
        _requiredWindow = inShape[inShape.length - 2];
        _numFeatures = inShape[inShape.length - 1];
      }
      if (outShape.isNotEmpty) {
        _numClasses = outShape.last;
      }

      bool dryOk = false;
      String? err;
      try {
        final winSize = _requiredWindow;
        final ch = _numFeatures;
        final outC = _numClasses;

        final input = List.generate(
          1,
          (_) => List.generate(
            winSize,
            (_) => List<double>.filled(ch, 0.0),
          ),
        );

        List<List<double>> output;
        if (outShape.length == 2 && outShape[0] == 1) {
          output = [List<double>.filled(outC, 0.0)];
        } else {
          output = [List<double>.filled(outC, 0.0)];
        }

        _interpreter!.run(input, output);
        dryOk = true;
      } catch (e) {
        err = e.toString();
      }

      _modelInfo = HarModelInfo(
        backend: 'tflite_flutter',
        inputShape: inShape,
        outputShape: outShape,
        inputDtype: inType,
        outputDtype: outType,
        dryRunOk: dryOk,
        error: err,
      );

      print('[HAR] Model loaded: backend=tflite_flutter '
          'in=$inShape($inType) out=$outShape($outType) '
          'dryRun=${dryOk ? "OK" : "FAIL"}${err != null ? " err=$err" : ""}');
      print('HAR model loaded successfully');
      return true;
    } catch (e) {
      _modelInfo = HarModelInfo(
        backend: 'tflite_flutter',
        inputShape: const [],
        outputShape: const [],
        inputDtype: 'unknown',
        outputDtype: 'unknown',
        dryRunOk: false,
        error: e.toString(),
      );
      print('Failed to load HAR model: $e');
      return false;
    }
  }

  static void addSample(double x, double y, double z) {
    _sampleBuffer.add([x, y, z]);

    if (_sampleBuffer.length > _requiredWindow) {
      _sampleBuffer.removeAt(0);
    }
  }

  static bool hasEnoughSamples() {
    return _sampleBuffer.length >= _requiredWindow;
  }

  static List<List<double>> _normalizeData(List<List<double>> data) {
    return data;
  }

  static String _labelFor(int idx) {
    if (idx >= 0 && idx < _activityLabels.length) {
      return _activityLabels[idx];
    }
    return 'Class $idx';
  }

  static String? predict() {
    if (_interpreter == null) {
      print('HAR model not initialized');
      return null;
    }

    if (!hasEnoughSamples()) {
      print('Not enough samples for prediction (${_sampleBuffer.length}/$_requiredWindow)');
      return null;
    }

    try {
      final int W = _requiredWindow;
      final int C = _numFeatures;
      final int K = _numClasses;

      List<List<double>> inputData = _sampleBuffer.sublist(_sampleBuffer.length - W);
      inputData = _normalizeData(inputData);

      if (inputData.isEmpty || inputData[0].length != C) {
        throw Exception('Input shape mismatch: got [${inputData.length}, ${inputData.isEmpty ? 0 : inputData[0].length}], expect [$W, $C]');
      }

      final input = <List<List<double>>>[
        List.generate(W, (j) => List<double>.from(inputData[j])),
      ];

      final output = <List<double>>[
        List<double>.filled(K, 0.0),
      ];

      _interpreter!.run(input, output);

      final List<double> probabilities = output[0].cast<double>();
      int predictedClass = 0;
      double maxProbability = probabilities[0];

      for (int i = 1; i < probabilities.length; i++) {
        if (probabilities[i] > maxProbability) {
          maxProbability = probabilities[i];
          predictedClass = i;
        }
      }

      if (maxProbability > 0.2) {
        final label = _labelFor(predictedClass);
        final msg = '$label (${(maxProbability * 100).toStringAsFixed(1)}%)';
        print('Predicted activity: $msg');
        return msg;
      } else {
        final label = _labelFor(predictedClass);
        return 'Uncertain: $label (${(maxProbability * 100).toStringAsFixed(1)}%)';
      }
    } catch (e) {
      print('Error during HAR prediction: $e');
      return null;
    }
  }

  static int getBufferSize() {
    return _sampleBuffer.length;
  }

  static int getRequiredWindow() => _requiredWindow;

  static void clearBuffer() {
    _sampleBuffer.clear();
  }

  static void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _sampleBuffer.clear();
  }
}