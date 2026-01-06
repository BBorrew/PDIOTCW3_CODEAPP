import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

class Task2Page extends StatefulWidget {
  const Task2Page({super.key});
  @override
  State<Task2Page> createState() => _Task2PageState();
}

class _Task2PageState extends State<Task2Page> {
  Interpreter? _itp;
  List<String> _labels = [];
  String _status = 'Loading...';

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      final itp = await Interpreter.fromAsset('assets/model_fp32_task2.tflite');
      final labelsRaw = await rootBundle.loadString('assets/labels_task2.txt');
      final labels = const LineSplitter()
          .convert(labelsRaw)
          .where((s) => s.trim().isNotEmpty)
          .toList();

      final in0 = itp.getInputTensor(0);
      final out0 = itp.getOutputTensor(0);
      debugPrint('[Task2] input=${in0.shape}, dtype=${in0.type}; output=${out0.shape}, dtype=${out0.type}');

      setState(() {
        _itp = itp;
        _labels = labels;
        _status = 'Model ready. Tap ▶ to run a test.';
      });
    } catch (e) {
      setState(() => _status = 'Load failed: $e');
    }
  }

  List<double> _softmax(List<double> x) {
    final m = x.reduce(math.max);
    final exps = x.map((v) => math.exp(v - m)).toList();
    final s = exps.fold(0.0, (a, b) => a + b) + 1e-9;
    return exps.map((v) => v / s).toList();
  }

  Future<void> _runOnce() async {
    final itp = _itp;
    if (itp == null) return;

    const win = 224;
    final input = [
      List.generate(win, (_) => [0.0, 0.0, 0.0]) // (224,3)
    ];
    final output = [
      List.filled(_labels.length, 0.0) // (1,4)
    ];

    setState(() => _status = 'Running inference...');
    itp.run(input, output);

    final raw = List<double>.from(output[0]);
    final sum = raw.fold(0.0, (a, b) => a + b);
    final probs = (sum > 0.98 && sum < 1.02) ? raw : _softmax(raw);

    var idx = 0; var best = -1e9;
    for (var i = 0; i < probs.length; i++) { if (probs[i] > best) { best = probs[i]; idx = i; } }

    setState(() => _status = 'Top-1: ${_labels[idx]}  (p=${best.toStringAsFixed(4)})');
  }

  @override
  void dispose() {
    _itp?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Task2 TFLite Smoke Test')),
      body: Center(child: Text(_status, style: const TextStyle(fontSize: 18))),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _runOnce,
        icon: const Icon(Icons.play_arrow),
        label: const Text('Run test'),
      ),
    );
  }
}
