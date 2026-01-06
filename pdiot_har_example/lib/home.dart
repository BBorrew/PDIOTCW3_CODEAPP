// lib/home.dart
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'settings.dart';
import 'utils.dart';
import 'ble.dart';
import 'globals.dart';
import 'dart:io';
import 'har_service.dart';
import 'task1_service.dart';
import 'task2_service.dart';

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});
  final String title;

  @override
  State<MyHomePage> createState() => MyHomePageState();
}

class MyHomePageState extends State<MyHomePage> {
  String accel = "";
  String batt_level = "";
  String recording_info = "Not recording";
  String predicted_activity = "HAR not initialized";

  bool _task1Ready = false;
  String _task1Status = "Task1 not initialized";
  String _task1Last = "—";

  bool _task2Ready = false;
  String _task2Status = "Task2 not initialized";
  String _task2Last = "—";
  bool _task2EvalRunning = false;

  int _task2Done = 0;
  int _task2Total = 0;
  String _task2CurCls = '';
  String _task2CurFile = '';

  var activities = [
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
  String selected_activity = "Standing";

  var social_signals = [
    'Normal',
    'Coughing',
    'Hyperventilating',
    'Talking',
    'Eating',
    'Singing',
    'Laughing'
  ];
  String selected_signal = "Normal";

  bool recording = false;
  bool received_packet = false;

  int recorded_samples = 0;
  DateTime? start_timestamp;

  String filename = "";

  @override
  void initState() {
    super.initState();
    _initializeHAR();
    _initializeTask1();
    _initializeTask2();
  }

  void _initializeHAR() async {
    final success = await HARService.initialize();
    setState(() {
      predicted_activity =
          success ? "HAR ready - waiting for data..." : "HAR initialization failed";
    });
  }

  Future<void> _initializeTask1() async {
    final ok = await Task1Service.initialize();
    setState(() {
      _task1Ready = ok;
      _task1Status = ok ? 'Task1 model loaded' : 'Task1 model load FAILED';
    });
  }

  Future<void> _initializeTask2() async {
    final ok = await Task2Service.initialize();
    setState(() {
      _task2Ready = ok;
      _task2Status = ok ? "Task2 model loaded" : "Task2 model load FAILED";
    });
  }

  Future<void> updateUI({
    required double x,
    required double y,
    required double z,
    int? batteryLevel,
    bool isCharging = false,
    required int respeckVersion,
    String? prediction,
    required int bufferSize,
  }) async {
    final needWin = HARService.getRequiredWindow();

    Task1Service.addSample(x, y, z);
    final r2 = Task2Service.feedAndPredict(x, y, z);
    if (r2 != null && mounted) {
      final label = r2['label'] as String;
      final conf = r2['confidence'] as double;
      updateTask2Label(label, conf);
    }

    setState(() {
      accel = "x=${x.toStringAsFixed(3)}, y=${y.toStringAsFixed(3)}, z=${z.toStringAsFixed(3)}";

      if (respeckVersion == 6) {
        batt_level = isCharging ? "Battery: $batteryLevel% (charging)" : "Battery: $batteryLevel%";
      } else {
        batt_level = "";
      }

      if (prediction != null) {
        predicted_activity = prediction;
      } else {
        predicted_activity = "Collecting data... ($bufferSize/$needWin samples)";
      }
    });

    if (Task1Service.hasEnough()) {
      final r1 = await Task1Service.predictContinuous();
      if (!mounted || r1 == null) return;
      final idx = r1['index'] as int;
      final conf = ((r1['confidence'] as double) * 100).toStringAsFixed(1);
      final name = (r1['label'] as String?) ?? 'class_$idx';
      setState(() {
        _task1Last = '$name ($conf%)';
      });
    }
  }

  @override
  void dispose() {
    HARService.dispose();
    Task1Service.dispose();
    Task2Service.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final infoHar = HARService.modelInfo;
    final infoT1 = Task1Service.modelInfo;
    final double progressValue =
        (_task2Total > 0) ? (_task2Done / _task2Total) : 0.0;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
        actions: [
          IconButton(
            tooltip: 'Task2 self check',
            icon: const Icon(Icons.science),
            onPressed: () async {
              try {
                await Navigator.pushNamed(context, '/task2');
              } catch (_) {
                if (mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Task2 page not registered')),
                  );
                }
              }
            },
          ),
        ],
      ),
      body: Center(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              ElevatedButton(
                  onPressed: () {
                    if (respeckUUID == null || respeckUUID == "") {
                      showToast("Please pair with a Respeck first");
                      return;
                    }
                    scanForRespeck(this);
                    print("finished scanning");
                  },
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.lightBlue),
                  child: const Text('Connect')),
              const SizedBox(height: 20),
              const Text(
                'Acceleration (g)',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              Text(accel),
              Text(batt_level),
              const SizedBox(height: 20),

              const Text(
                'Real-time Activity Recognition',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              Container(
                padding: const EdgeInsets.all(10),
                margin: const EdgeInsets.symmetric(horizontal: 20),
                decoration: BoxDecoration(
                  color: Colors.blue.shade50,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.blue.shade200),
                ),
                child: Text(
                  predicted_activity,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w500,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
              if (infoHar != null) ...[
                const SizedBox(height: 12),
                _modelInfoCard(
                  ok: infoHar.dryRunOk,
                  titleOk: 'HAR Model status: OK',
                  titleFail: 'HAR Model status: FAILED',
                  backend: infoHar.backend,
                  input: '${infoHar.inputShape} (${infoHar.inputDtype})',
                  output: '${infoHar.outputShape} (${infoHar.outputDtype})',
                  error: infoHar.error,
                ),
              ],

              const SizedBox(height: 12),
              _task1Card(infoT1),

              const SizedBox(height: 12),
              _task2Card(progressValue),

              const SizedBox(height: 20),
              const Text(
                'Activity',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              DropdownButton(
                value: selected_activity,
                items: activities.map((String items) {
                  return DropdownMenuItem(value: items, child: Text(items));
                }).toList(),
                onChanged: (String? newValue) {
                  setState(() {
                    selected_activity = newValue!;
                  });
                },
              ),
              const SizedBox(height: 10),
              const Text(
                'Social signal',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              DropdownButton(
                value: selected_signal,
                items: social_signals.map((String items) {
                  return DropdownMenuItem(value: items, child: Text(items));
                }).toList(),
                onChanged: (String? newValue) {
                  setState(() {
                    selected_signal = newValue!;
                  });
                },
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                  onPressed: record,
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.lightGreen),
                  child: const Text('Start recording')),
              const SizedBox(height: 20),
              const Text(
                "Recording status",
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              Text(recording_info),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20.0),
                child: Text(filename),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                  onPressed: () {
                    if (!recording) return;
                    recording = false;
                    showToast("Recording stopped");
                  },
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.red[200]!),
                  child: const Text('Stop recording')),
              const SizedBox(height: 20),
              ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => const SettingsPage()),
                    );
                  },
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.grey),
                  child: const Text('Settings')),
            ],
          ),
        ),
      ),
    );
  }

  Widget _modelInfoCard({
    required bool ok,
    required String titleOk,
    required String titleFail,
    required String backend,
    required String input,
    required String output,
    String? error,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20.0),
      child: Card(
        elevation: 0,
        color: ok ? Colors.green.shade50 : Colors.red.shade50,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
          side: BorderSide(color: ok ? Colors.green.shade200 : Colors.red.shade200),
        ),
        child: Padding(
          padding: const EdgeInsets.all(10.0),
          child: Column(
            children: [
              Row(
                children: [
                  Icon(ok ? Icons.check_circle : Icons.error,
                      color: ok ? Colors.green : Colors.red),
                  const SizedBox(width: 8),
                  Text(
                    ok ? titleOk : titleFail,
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: ok ? Colors.green.shade800 : Colors.red.shade800,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 6),
              _kv('Backend', backend),
              _kv('Input', input),
              _kv('Output', output),
              if (!ok && error != null) _kv('Error', error),
            ],
          ),
        ),
      ),
    );
  }

  Widget _task1Card(TfliteModelInfo infoT1) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20.0),
      child: Card(
        elevation: 0,
        color: _task1Ready ? Colors.green.shade50 : Colors.red.shade50,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
          side: BorderSide(
            color: _task1Ready ? Colors.green.shade200 : Colors.red.shade200,
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(10.0),
          child: Column(
            children: [
              Row(
                children: [
                  Icon(_task1Ready ? Icons.check_circle : Icons.error,
                      color: _task1Ready ? Colors.green : Colors.red),
                  const SizedBox(width: 8),
                  Text(
                    _task1Ready ? 'Task1 Model status: OK' : 'Task1 Model status: FAILED',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: _task1Ready ? Colors.green.shade800 : Colors.red.shade800,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 6),
              _kv('Status', _task1Status),
              _kv('Last prediction', _task1Last),
              const SizedBox(height: 6),
              _kv('Backend', infoT1.backend),
              _kv('Input', '${infoT1.inputShape} (${infoT1.inputDtype})'),
              _kv('Output', '${infoT1.outputShape} (${infoT1.outputDtype})'),
              if (!_task1Ready && infoT1.error != null) _kv('Error', infoT1.error!),
            ],
          ),
        ),
      ),
    );
  }

  Widget _task2Card(double progressValue) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20.0),
      child: Card(
        elevation: 0,
        color: _task2Ready ? Colors.green.shade50 : Colors.red.shade50,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
          side: BorderSide(
            color: _task2Ready ? Colors.green.shade200 : Colors.red.shade200,
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(10.0),
          child: Column(
            children: [
              Row(
                children: [
                  Icon(_task2Ready ? Icons.check_circle : Icons.error,
                      color: _task2Ready ? Colors.green : Colors.red),
                  const SizedBox(width: 8),
                  Text(
                    _task2Ready ? 'Task2 Model status: OK' : 'Task2 Model status: FAILED',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: _task2Ready ? Colors.green.shade800 : Colors.red.shade800,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 6),
              _kv('Status', _task2Status),
              _kv('Last prediction', _task2Last),

              if (_task2EvalRunning) ...[
                const SizedBox(height: 10),
                SizedBox(
                  width: double.infinity,
                  child: LinearProgressIndicator(
                    value: (_task2Total > 0) ? progressValue.clamp(0.0, 1.0) : null,
                    minHeight: 6,
                  ),
                ),
                const SizedBox(height: 6),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    'Progress: $_task2Done / $_task2Total'
                    '${_task2CurCls.isNotEmpty ? '  ($_task2CurCls)' : ''}',
                    style: const TextStyle(fontSize: 12),
                  ),
                ),
                const SizedBox(height: 2),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    _task2CurFile,
                    overflow: TextOverflow.ellipsis,
                    maxLines: 1,
                    style: const TextStyle(fontSize: 12, color: Colors.black54),
                  ),
                ),
              ],

              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: (_task2Ready && !_task2EvalRunning)
                          ? () async {
                              setState(() {
                                _task2EvalRunning = true;
                                _task2Status = "Evaluating dataset…";
                                _task2Done = 0;
                                _task2Total = 0;
                                _task2CurCls = '';
                                _task2CurFile = '';
                              });
                              try {
                                final res = await Task2Service.selfTest(
                                  concurrency: 2,
                                  verbose: false,
                                  onProgress: (done, total, cls, file) {
                                    if (!mounted) return;
                                    setState(() {
                                      _task2Done = done;
                                      _task2Total = total;
                                      _task2CurCls = cls;
                                      _task2CurFile = file;
                                    });
                                  },
                                );

                                if (!mounted) return;

                                if (res.containsKey('error')) {
                                  showToast('Task2 eval error: ${res['error']}');
                                  setState(() {
                                    _task2Status = "Eval error";
                                  });
                                  return;
                                }

                                final Map<String, dynamic> per =
                                    (res['per_class'] as Map).map(
                                        (k, v) => MapEntry(k.toString(), (v as num).toDouble()),
                                    );
                                final double macro = (res['macro_acc'] as num).toDouble();
                                final int total = (res['total_windows'] as num).toInt();
                                final Map<String, dynamic> counts =
                                    (res['counts'] as Map).map(
                                        (k, v) => MapEntry(k.toString(), (v as num).toInt()),
                                    );

                                setState(() {
                                  _task2Last =
                                      'Macro ${(macro * 100).toStringAsFixed(1)}% ($total windows)';
                                  _task2Status = "Eval finished";
                                });

                                await showDialog<void>(
                                  context: context,
                                  builder: (ctx) {
                                    String line(String k) {
                                      final acc =
                                          ((per[k] ?? 0.0) * 100).toStringAsFixed(1);
                                      final n = counts[k] ?? 0;
                                      return '$k: $acc%  (N=$n)';
                                    }
                                    return AlertDialog(
                                      title: const Text('Task2 数据集评测'),
                                      content: Column(
                                        mainAxisSize: MainAxisSize.min,
                                        crossAxisAlignment: CrossAxisAlignment.start,
                                        children: [
                                          Text(line('breathing')),
                                          Text(line('coughing')),
                                          Text(line('hyperventilating')),
                                          Text(line('other')),
                                          const SizedBox(height: 8),
                                          Text(
                                            'Macro Acc: ${(macro * 100).toStringAsFixed(1)}%  Total: $total windows',
                                            style: const TextStyle(
                                                fontWeight: FontWeight.bold),
                                          ),
                                        ],
                                      ),
                                      actions: [
                                        TextButton(
                                          onPressed: () => Navigator.pop(ctx),
                                          child: const Text('OK'),
                                        ),
                                      ],
                                    );
                                  },
                                );
                              } catch (e) {
                                if (mounted) {
                                  showToast('Task2 self-test exception: $e');
                                  setState(() {
                                    _task2Status = "Eval exception";
                                  });
                                }
                              } finally {
                                if (mounted) {
                                  setState(() {
                                    _task2EvalRunning = false;
                                  });
                                }
                              }
                            }
                          : null,
                      icon: _task2EvalRunning
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.science),
                      label: Text(_task2EvalRunning ? 'testing…' : 'test 2526 here'),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 4),
              Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  'Realtime: auto from BLE stream',
                  style: const TextStyle(fontSize: 12, color: Colors.black54),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _kv(String k, String v) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
            width: 110,
            child: Text('$k:', style: const TextStyle(fontWeight: FontWeight.w600))),
        Expanded(child: Text(v, overflow: TextOverflow.ellipsis, maxLines: 1)),
      ],
    );
  }

  void updateTask2Label(String label, double conf) {
    setState(() {
      _task2Last = '$label (${(conf * 100).toStringAsFixed(1)}%)';
      _task2Status = 'Task2 realtime running';
    });
  }

  void record() async {
    if (!received_packet) {
      showToast("Please connect to a Respeck first");
      return;
    }
    if (recording) {
      showToast("Already recording");
      return;
    }

    recorded_samples = 0;
    final now = DateTime.now().toUtc();
    start_timestamp = now;
    final formattedDate =
        '${DateFormat('yyyy-MM-dd').format(now)}T${DateFormat('kkmmss').format(now)}Z';
    filename =
        'PDIOT_${subjectID}_${sentenceToCamelCase(selected_activity)}_${sentenceToCamelCase(selected_signal)}_${formattedDate}_${respeckUUID?.replaceAll(":", "")}.csv';
    print(filename);

    csvFile = File('${storageFolder?.path}/$filename');
    await csvFile.writeAsString(
      "receivedPhoneTimestamp,respeckTimestamp,packetSeqNum,sampleSeqNum,accelX,accelY,accelZ\n",
      flush: true,
    );
    showToast("Recording started..");
    recording = true;
  }
}