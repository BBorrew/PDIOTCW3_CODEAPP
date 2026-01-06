// lib/task2_service.dart
import 'dart:math' as math;
import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';


typedef Task2Progress = void Function(int, int, String, String);

class Task2Service {
  static Interpreter? _itp;

  
  static int _win = 300;
  
  static int _feat = 3;
  
  static bool _channelsFirst = false;

  static int get win => _win;
  static int get feat => _feat;
  static bool get channelsFirst => _channelsFirst;

  static const int numClasses = 4;
  static const List<String> labels = [
    'breathing', 'coughing', 'hyperventilating', 'other'
  ];

  
  static const String evalAssetRoot = 'assets/social_signal';
  static const String evalIndexAsset = 'assets/social_signal/index.json';

  
  static final List<List<double>> _buf = [];

 
  static Future<bool> initialize() async {
    try {
      _itp?.close();
      _itp = await Interpreter.fromAsset('assets/best_fold1_minvalacc.tflite');
      _itp!.allocateTensors();

      final in0 = _itp!.getInputTensor(0);
      final out0 = _itp!.getOutputTensor(0);

      
      final s = List<int>.from(in0.shape);
      if (s.length != 3 || s.first != 1) {
        throw Exception('Unexpected input shape: ${in0.shape}');
      }
      final d1 = s[1], d2 = s[2];
      if (d1 == 3 && d2 >= 8) {
        // [1, C, T]
        _channelsFirst = true;
        _feat = 3;
        _win = d2;
      } else if (d2 == 3 && d1 >= 8) {
        // [1, T, C]
        _channelsFirst = false;
        _feat = 3;
        _win = d1;
      } else {
    
        _channelsFirst = d1 < d2;
        _win = math.max(d1, d2);
        _feat = math.min(d1, d2);
      }

     
      final fakeInput = _channelsFirst
          // [1, C, T]
          ? List.generate(
              1,
              (_) => List.generate(
                _feat,
                (_) => List.filled(_win, 0.0),
              ),
            )
          // [1, T, C]
          : List.generate(
              1,
              (_) => List.generate(
                _win,
                (_) => List.filled(_feat, 0.0),
              ),
            );

      final fakeOutput =
          List.generate(1, (_) => List.filled(out0.shape.last, 0.0));
      _itp!.run(fakeInput, fakeOutput);

      print('[Task2] Model loaded: asset=assets/model_fp32_task2.tflite '
          'in=${in0.shape}(${in0.type}) out=${out0.shape}(${out0.type}) '
          'layout=${_channelsFirst ? 'NCT' : 'NTC'} win=$_win feat=$_feat dryRun=OK');

      return true;
    } catch (e) {
      print('[Task2] failed to load: $e');
      return false;
    }
  }

  static void dispose() {
    _itp?.close();
    _itp = null;
    _buf.clear();
  }

  
  static void addSample(double x, double y, double z) {
    _buf.add([x, y, z]);
    if (_buf.length > _win) _buf.removeAt(0);
  }

  static bool hasEnough() => _buf.length >= _win;

  static List<List<double>> _zscore(List<List<double>> w) {
    final mu = List<double>.filled(_feat, 0.0);
    final sigma = List<double>.filled(_feat, 0.0);

    for (var j = 0; j < _feat; j++) {
      double s = 0;
      for (var i = 0; i < _win; i++) s += w[i][j];
      mu[j] = s / _win;
    }
    for (var j = 0; j < _feat; j++) {
      double v = 0;
      for (var i = 0; i < _win; i++) {
        final d = w[i][j] - mu[j];
        v += d * d;
      }
      sigma[j] = math.sqrt(v / _win);
      if (sigma[j] < 1e-6) sigma[j] = 1.0;
    }
    final out =
        List<List<double>>.generate(_win, (_) => List<double>.filled(_feat, 0.0));
    for (var i = 0; i < _win; i++) {
      for (var j = 0; j < _feat; j++) {
        out[i][j] = (w[i][j] - mu[j]) / sigma[j];
      }
    }
    return out;
  }

  static Map<String, dynamic>? predict() {
    if (_itp == null) return null;
    if (!hasEnough()) return null;

    final window = _buf.sublist(_buf.length - _win);
    final norm = _zscore(window);

  
    Object input;
    if (_channelsFirst) {
      // [1, C, T]
      final x = List.generate(
          1, (_) => List.generate(_feat, (_) => List.filled(_win, 0.0)));
      for (int t = 0; t < _win; t++) {
        for (int c = 0; c < _feat; c++) {
          x[0][c][t] = norm[t][c];
        }
      }
      input = x;
    } else {
      // [1, T, C]
      final x = List.generate(
          1, (_) => List.generate(_win, (_) => List.filled(_feat, 0.0)));
      for (int t = 0; t < _win; t++) {
        for (int c = 0; c < _feat; c++) {
          x[0][t][c] = norm[t][c];
        }
      }
      input = x;
    }

    final output = List.generate(1, (_) => List.filled(numClasses, 0.0));

    try {
      _itp!.run(input, output);
      final probs =
          (output[0] as List).map((e) => (e as num).toDouble()).toList();
      final sm = _softmax(probs);

      var bestIdx = 0;
      var bestP = sm[0];
      for (var i = 1; i < sm.length; i++) {
        if (sm[i] > bestP) {
          bestP = sm[i];
          bestIdx = i;
        }
      }
      return {
        'index': bestIdx,
        'label': labels[bestIdx],
        'confidence': bestP,
      };
    } catch (e) {
      print('[Task2] inference error: $e');
      return null;
    }
  }

  
  static Map<String, dynamic>? feedAndPredict(
      double x, double y, double z) {
    _buf.add([x, y, z]);
    if (_buf.length > _win) _buf.removeAt(0);

    if (_itp == null) return null;
    if (_buf.length < _win) return null;

    final window = _buf.sublist(_buf.length - _win);
    return _predictOnWindowRealtime(window);
  }

  static Map<String, dynamic>? _predictOnWindowRealtime(
      List<List<double>> window) {
    if (_itp == null) return null;
    if (window.length != _win) return null;

    try {
      final norm = _zscore(window);

      Object input;
      if (_channelsFirst) {
        // [1, C, T]
        final x = List.generate(
            1, (_) => List.generate(_feat, (_) => List.filled(_win, 0.0)));
        for (int t = 0; t < _win; t++) {
          for (int c = 0; c < _feat; c++) {
            x[0][c][t] = norm[t][c];
          }
        }
        input = x;
      } else {
        // [1, T, C]
        final x = List.generate(
            1, (_) => List.generate(_win, (_) => List.filled(_feat, 0.0)));
        for (int t = 0; t < _win; t++) {
          for (int c = 0; c < _feat; c++) {
            x[0][t][c] = norm[t][c];
          }
        }
        input = x;
      }

      final output = List.generate(1, (_) => List.filled(numClasses, 0.0));

      _itp!.run(input, output);
      final probs =
          (output[0] as List).map((e) => (e as num).toDouble()).toList();
      final sm = _softmax(probs);

      var bestIdx = 0;
      var bestP = sm[0];
      for (var i = 1; i < sm.length; i++) {
        if (sm[i] > bestP) {
          bestP = sm[i];
          bestIdx = i;
        }
      }
      return {
        'index': bestIdx,
        'label': labels[bestIdx],
        'confidence': bestP,
      };
    } catch (e) {
      print('[Task2] realtime inference error: $e');
      return null;
    }
  }

  static List<double> _softmax(List<double> logits) {
    final mx = logits.reduce(math.max);
    var sum = 0.0;
    final exps = logits
        .map((p) {
          final e = math.exp(p - mx);
          sum += e;
          return e;
        })
        .toList();
    return exps.map((e) => e / (sum + 1e-9)).toList();
  }

  static void clearBuffer() => _buf.clear();

 
  static Future<Map<String, dynamic>> selfTest({
    int concurrency = 1,
    bool verbose = false,
    Task2Progress? onProgress,
  }) async {
    return await evaluateFromAssets(
      concurrency: concurrency,
      verbose: verbose,
      onProgress: onProgress,
    );
  }

  
  static Future<Map<String, dynamic>> evaluateFromAssets({
    int concurrency = 1,
    bool verbose = false,
    Task2Progress? onProgress,
  }) async {
    if (_itp == null) {
      return {"error": "model_not_initialized"};
    }

    
    try {
      final manifestText = await rootBundle.loadString('AssetManifest.json');
      if (!manifestText.contains(evalIndexAsset)) {
        print(
            '[Task2][eval][DEBUG] $evalIndexAsset not found in AssetManifest');
      } else {
        print('[Task2][eval][DEBUG] $evalIndexAsset hit in AssetManifest');
      }
      final Map<String, dynamic> mani = json.decode(manifestText);
      final keys = mani.keys
          .where((k) => k.toString().startsWith('$evalAssetRoot/'))
          .toList()
        ..sort();
      print(
          '[Task2][eval][DEBUG] Number of assets under $evalAssetRoot: ${keys.length}');
      for (int i = 0; i < (keys.length < 8 ? keys.length : 8); i++) {
        print('   - ${keys[i]}');
      }
    } catch (e) {
      print('[Task2][eval][DEBUG] Failed to read AssetManifest: $e');
    }

    Map<String, dynamic> idx;
    try {
      final txt = await rootBundle.loadString(evalIndexAsset);
      idx = json.decode(txt) as Map<String, dynamic>;
    } catch (e) {
      return {"error": "index_json_missing_or_invalid: $e"};
    }

    
    final Map<String, List<String>> filesByClass = {};
    for (final cls in labels) {
      final List<dynamic>? arr = idx[cls] as List<dynamic>?;
      filesByClass[cls] =
          arr?.map((e) => e.toString()).toList() ?? <String>[];
    }

    final List<MapEntry<String, String>> all = [];
    for (final cls in labels) {
      for (final rel in filesByClass[cls]!) {
        all.add(MapEntry(cls, rel));
      }
    }
    final totalFiles = all.length;
    int doneFiles = 0;

   
    final Map<String, int> totalByClass = {for (var c in labels) c: 0};
    final Map<String, int> correctByClass = {for (var c in labels) c: 0};

    
    final int k = (concurrency <= 0) ? 1 : concurrency;
    for (int i = 0; i < all.length; i += k) {
      final chunk =
          all.sublist(i, (i + k > all.length) ? all.length : i + k);
      for (final kv in chunk) {
        final cls = kv.key;
        final relPath = kv.value;
        final assetPath = '$evalAssetRoot/$relPath';

        List<List<double>> seq;
        try {
          final csvText = await rootBundle.loadString(assetPath);
          seq = _parseCsvXYZ(csvText);
        } catch (e) {
          if (verbose) {
            print(
                '[Task2][eval] skip (read-error): $assetPath -> $e');
          }
          doneFiles++;
          onProgress?.call(
              doneFiles, totalFiles, cls, relPath);
          continue;
        }

        if (seq.length < _win) {
          if (verbose) {
            print(
                '[Task2][eval] skip (too short ${seq.length}): $assetPath');
          }
          doneFiles++;
          onProgress?.call(
              doneFiles, totalFiles, cls, relPath);
          continue;
        }

     
        for (int start = 0; start + _win <= seq.length; start += _win) {
          final window = seq.sublist(start, start + _win);
          final ok =
              _predictOneWindow(window, expectClass: cls);
          totalByClass[cls] = totalByClass[cls]! + 1;
          if (ok) correctByClass[cls] = correctByClass[cls]! + 1;
        }

        doneFiles++;
        onProgress?.call(
            doneFiles, totalFiles, cls, relPath);
      }
      await Future<void>.delayed(
          const Duration(milliseconds: 1));
    }

    
    final Map<String, double> perClass = {};
    double macro = 0.0;
    int used = 0;
    int totalWindows = 0;
    for (final cls in labels) {
      final t = totalByClass[cls]!;
      totalWindows += t;
      final c = correctByClass[cls]!;
      final acc = t > 0 ? c / t : 0.0;
      perClass[cls] = acc;
      if (t > 0) {
        macro += acc;
        used += 1;
      }
    }
    final macroAcc = used > 0 ? macro / used : 0.0;

    final result = {
      "per_class": perClass,
      "macro_acc": macroAcc,
      "counts": totalByClass,
      "total_windows": totalWindows,
    };
    print('[Task2][eval] result: $result');
    return result;
  }

  static bool _predictOneWindow(
    List<List<double>> window, {
    required String expectClass,
  }) {
    try {
      final norm = _zscore(window);

      Object input;
      if (_channelsFirst) {
        // [1, C, T]
        final x = List.generate(
            1, (_) => List.generate(_feat, (_) => List.filled(_win, 0.0)));
        for (int t = 0; t < _win; t++) {
          for (int c = 0; c < _feat; c++) {
            x[0][c][t] = norm[t][c];
          }
        }
        input = x;
      } else {
        // [1, T, C]
        final x = List.generate(
            1, (_) => List.generate(_win, (_) => List.filled(_feat, 0.0)));
        for (int t = 0; t < _win; t++) {
          for (int c = 0; c < _feat; c++) {
            x[0][t][c] = norm[t][c];
          }
        }
        input = x;
      }

      final output = List.generate(1, (_) => List.filled(numClasses, 0.0));
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
      return labels[argmax] == expectClass;
    } catch (e) {
      print('[Task2][eval] window inference error: $e');
      return false;
    }
  }

  
  static List<List<double>> _parseCsvXYZ(String csv) {
    final lines = const LineSplitter().convert(csv);
    final out = <List<double>>[];
    for (int i = 0; i < lines.length; i++) {
      final line = lines[i].trim();
      if (line.isEmpty) continue;
      if (i == 0 && _looksLikeHeader(line)) continue;

      final parts = line
          .split(RegExp(r'[,\s;]+'))
          .where((s) => s.isNotEmpty)
          .toList();
      if (parts.length < 3) continue;
      try {
        final n = parts.length;
        final x = double.parse(parts[n - 3]);
        final y = double.parse(parts[n - 2]);
        final z = double.parse(parts[n - 1]);
        out.add([x, y, z]);
      } catch (_) {
        // Skip malformed line
      }
    }
    return out;
  }

  static bool _looksLikeHeader(String line) {
    final low = line.toLowerCase();
    return low.contains('acc') ||
        (low.contains('x') && low.contains('y') && low.contains('z'));
  }
}