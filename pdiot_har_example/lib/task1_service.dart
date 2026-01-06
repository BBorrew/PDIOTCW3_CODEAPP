import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

class TfliteModelInfo {
  final String backend;
  final String inputShape;
  final String outputShape;
  final String inputDtype;
  final String outputDtype;
  final bool dryRunOk;
  final String? error;

  const TfliteModelInfo({
    required this.backend,
    required this.inputShape,
    required this.outputShape,
    required this.inputDtype,
    required this.outputDtype,
    required this.dryRunOk,
    this.error,
  });
}

class Task1Service {
  static const String kModelA = 'assets/model_fp32_task1.tflite';
  static const String kModelB = 'model_fp32_task1.tflite';

  static const String kLabelsA = 'assets/task1/labels.txt';
  static const String kLabelsB = 'task1/labels.txt';
  static const String kNormA = 'assets/task1/norm_stats.json';
  static const String kNormB = 'task1/norm_stats.json';

  static int _win = 128;
  static int _feat = 3;
  static bool _channelsFirst = false;

  static List<String> _labels = <String>[];
  static List<double>? _mu;
  static List<double>? _sigma;
  static bool get hasGlobalNorm => _mu != null && _sigma != null;

  static Interpreter? _itp;
  static bool _busy = false;
  static bool _ready = false;
  static bool get isReady => _ready;

  static String _backend = 'tflite_flutter';
  static String _inputShape = '';
  static String _outputShape = '';
  static String _inputDtype = '';
  static String _outputDtype = '';
  static bool _dryRunOk = false;
  static String? _lastError;

  static TfliteModelInfo get modelInfo => TfliteModelInfo(
        backend: _backend,
        inputShape: _inputShape,
        outputShape: _outputShape,
        inputDtype: _inputDtype,
        outputDtype: _outputDtype,
        dryRunOk: _dryRunOk,
        error: _lastError,
      );

  static final List<List<double>> _buf = [];
  static int getBufferLen() => _buf.length;
  static int getRequiredWindow() => _win;
  static List<String> get labels => _labels;

  static Future<bool> initialize() async {
    _resetStatus();

    for (final path in <String>[kModelA, kModelB]) {
      try {
        _itp?.close();
        _itp = await Interpreter.fromAsset(path);
        _itp!.allocateTensors();

        final in0 = _itp!.getInputTensor(0);
        final out0 = _itp!.getOutputTensor(0);

        _inputShape = in0.shape.toString();
        _outputShape = out0.shape.toString();
        _inputDtype = in0.type.toString();
        _outputDtype = out0.type.toString();

        final s = List<int>.from(in0.shape);
        if (s.length != 3 || s.first != 1) {
          throw Exception('Unexpected input shape: ${in0.shape}');
        }
        final d1 = s[1], d2 = s[2];
        if (d1 == 3 && d2 >= 8) {
          _channelsFirst = true;
          _feat = 3;
          _win = d2;
        } else if (d2 == 3 && d1 >= 8) {
          _channelsFirst = false;
          _feat = 3;
          _win = d1;
        } else {
          _channelsFirst = d1 < d2;
          _win = math.max(d1, d2);
          _feat = math.min(d1, d2);
        }

        final fake = _channelsFirst
            ? List.generate(
                1, (_) => List.generate(_feat, (_) => List.filled(_win, 0.0)))
            : List.generate(
                1, (_) => List.generate(_win, (_) => List.filled(_feat, 0.0)));
        final out =
            List.generate(1, (_) => List.filled(out0.shape.last, 0.0));
        _itp!.run(fake, out);
        _dryRunOk = true;

        print(
            '[Task1] Model loaded: asset=$path in=$_inputShape($_inputDtype) out=$_outputShape($_outputDtype) layout=${_channelsFirst ? 'NCT' : 'NTC'} win=$_win feat=$_feat dryRun=OK');

        await _loadLabels();
        await _loadNormStats();

        _ready = true;
        return true;
      } catch (e) {
        _lastError = '$e';
        print('[Task1] load failed with "$path": $e');
      }
    }
    _ready = false;
    return false;
  }

  static Future<void> _loadLabels() async {
    for (final p in <String>[kLabelsA, kLabelsB]) {
      try {
        final txt = await rootBundle.loadString(p);
        final lines = txt
            .split(RegExp(r'\r?\n'))
            .map((e) => e.trim())
            .where((e) => e.isNotEmpty)
            .toList();
        if (lines.isNotEmpty) {
          _labels = lines;
          print('[Task1] labels loaded (${_labels.length}) from $p');
          return;
        }
      } catch (_) {}
    }
    if (_labels.isEmpty) {
      _labels = List<String>.generate(
          _itp?.getOutputTensor(0).shape.last ?? 4, (i) => 'class_$i');
      print('[Task1] labels not found, fallback: $_labels');
    }
  }

  static Future<void> _loadNormStats() async {
    for (final p in <String>[kNormA, kNormB]) {
      try {
        final js = json.decode(await rootBundle.loadString(p)) as Map<String, dynamic>;
        final mu =
            (js['mu'] as List).map((e) => (e as num).toDouble()).toList();
        final sigma =
            (js['sigma'] as List).map((e) => (e as num).toDouble()).toList();
        if (mu.length == sigma.length && mu.isNotEmpty) {
          _mu = mu;
          _sigma = sigma;
          if (js.containsKey('window_size')) {
            final w = (js['window_size'] as num).toInt();
            if (w > 8) _win = w;
          }
          if (js.containsKey('feat_channels')) {
            final c = (js['feat_channels'] as num).toInt();
            if (c >= 1) _feat = c;
          }
          print('[Task1] norm_stats loaded from $p (win=$_win feat=$_feat)');
          return;
        }
      } catch (_) {}
    }
    print('[Task1] norm_stats not found, will use window z-score.');
  }

  static void dispose() {
    _itp?.close();
    _itp = null;
    _buf.clear();
    _ready = false;
  }

  static void _resetStatus() {
    _backend = 'tflite_flutter';
    _inputShape = '';
    _outputShape = '';
    _inputDtype = '';
    _outputDtype = '';
    _dryRunOk = false;
    _lastError = null;
    _labels = <String>[];
    _mu = null;
    _sigma = null;
    _win = 128;
    _feat = 3;
    _channelsFirst = false;
  }

  static void addSample(double x, double y, double z) {
    _buf.add([x, y, z]);
    if (_buf.length > _win) _buf.removeAt(0);
    if (_buf.length % 32 == 0) {
      print('[Task1][DEBUG] buf=${_buf.length}/$_win');
    }
  }

  static void addBatch(List<List<double>> samples) {
    for (final v in samples) {
      if (v.isEmpty) continue;
      final x = v.isNotEmpty ? v[0] : 0.0;
      final y = v.length > 1 ? v[1] : 0.0;
      final z = v.length > 2 ? v[2] : 0.0;
      _buf.add([x, y, z]);
      if (_buf.length > _win) _buf.removeAt(0);
    }
    if (_buf.length % 32 == 0 || _buf.length >= _win) {
      print('[Task1][DEBUG] buf=${_buf.length}/$_win');
    }
  }

  static bool hasEnough() => _buf.length >= _win;

  static Future<Map<String, dynamic>?> predictContinuous() async {
    if (_itp == null || !_ready) return null;
    if (!hasEnough()) return null;
    if (_busy) return null;
    _busy = true;

    try {
      final window = _buf.sublist(_buf.length - _win);
      final List<List<double>> norm = hasGlobalNorm
          ? _applyGlobalNorm(window)
          : _zscore(window);

      Object input;
      if (_channelsFirst) {
        // [1, C, T]
        final x = List.generate(
            1, (_) => List.generate(_feat, (_) => List.filled(_win, 0.0)));
        for (int t = 0; t < _win; t++) {
          for (int c = 0; c < _feat; c++) x[0][c][t] = norm[t][c];
        }
        input = x;
      } else {
        // [1, T, C]
        final x = List.generate(
            1, (_) => List.generate(_win, (_) => List.filled(_feat, 0.0)));
        for (int t = 0; t < _win; t++) {
          for (int c = 0; c < _feat; c++) x[0][t][c] = norm[t][c];
        }
        input = x;
      }

      final out0 = _itp!.getOutputTensor(0);
      final outSize = out0.shape.last;
      final output = List.generate(1, (_) => List.filled(outSize, 0.0));

      _itp!.run(input, output);

      final probs =
          (output[0] as List).map((e) => (e as num).toDouble()).toList();
      final sm = _softmax(probs);

      int argmax = 0;
      double best = sm[0];
      for (int i = 1; i < sm.length; i++) {
        if (sm[i] > best) {
          best = sm[i];
          argmax = i;
        }
      }
      final label =
          (argmax < _labels.length) ? _labels[argmax] : 'class_$argmax';
      return {'index': argmax, 'label': label, 'confidence': best, 'probs': sm};
    } catch (e) {
      _lastError = '[Task1] inference error: $e';
      print(_lastError);
      return null;
    } finally {
      _busy = false;
    }
  }

  static List<List<double>> _applyGlobalNorm(List<List<double>> w) {
    final mu = _mu!, sg = _sigma!;
    final out =
        List<List<double>>.generate(_win, (_) => List<double>.filled(_feat, 0.0));
    for (int t = 0; t < _win; t++) {
      for (int c = 0; c < _feat; c++) {
        final s = (c < sg.length && sg[c].abs() > 1e-9) ? sg[c] : 1.0;
        final m = (c < mu.length) ? mu[c] : 0.0;
        out[t][c] = (w[t][c] - m) / s;
      }
    }
    return out;
  }

  static List<List<double>> _zscore(List<List<double>> w) {
    final mu = List<double>.filled(_feat, 0.0);
    final sigma = List<double>.filled(_feat, 0.0);
    for (int c = 0; c < _feat; c++) {
      double s = 0;
      for (int t = 0; t < _win; t++) s += w[t][c];
      mu[c] = s / _win;
    }
    for (int c = 0; c < _feat; c++) {
      double v = 0;
      for (int t = 0; t < _win; t++) {
        final d = w[t][c] - mu[c];
        v += d * d;
      }
      sigma[c] = math.sqrt(v / _win);
      if (sigma[c] < 1e-6) sigma[c] = 1.0;
    }
    final out =
        List<List<double>>.generate(_win, (_) => List<double>.filled(_feat, 0.0));
    for (int t = 0; t < _win; t++) {
      for (int c = 0; c < _feat; c++) out[t][c] = (w[t][c] - mu[c]) / sigma[c];
    }
    return out;
  }

  static List<double> _softmax(List<double> logits) {
    final mx = logits.reduce(math.max);
    double sum = 0.0;
    final exps = logits.map((p) {
      final e = math.exp(p - mx);
      sum += e;
      return e;
    }).toList();
    return exps.map((e) => e / (sum + 1e-9)).toList();
  }
}