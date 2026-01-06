// lib/ble.dart
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'dart:io';
import 'dart:typed_data';
import 'utils.dart';
import 'home.dart';
import 'globals.dart';
import 'har_service.dart';
import 'task1_service.dart';
import 'task2_service.dart';

Future<void> scanForRespeck(MyHomePageState ui) async {
  BluetoothDevice? respeck;
  int respeckVersion = 0;

  showToast("Searching for Respeck $respeckUUID...");

  var subscription = FlutterBluePlus.onScanResults.listen(
    (results) {
      if (results.isNotEmpty) {
        for (ScanResult r in results) {
          if (respeck == null && r.device.remoteId.str == respeckUUID) {
            if (r.advertisementData.advName == "Res6AL") {
              respeck = r.device;
              respeckVersion = 6;
              fwString = "6AL";
            } else if (r.advertisementData.advName == "ResV5i") {
              respeck = r.device;
              respeckVersion = 5;
              fwString = "5i";
            } else {
              showToast("Respeck firmware is too old");
            }
          }
        }
      }
    },
    onError: (e) => print(e),
  );

  FlutterBluePlus.cancelWhenScanComplete(subscription);

  await FlutterBluePlus.adapterState
      .where((val) => val == BluetoothAdapterState.on)
      .first;

  await FlutterBluePlus.startScan(timeout: const Duration(seconds: 5));
  await FlutterBluePlus.isScanning.where((val) => val == false).first;
  print("end of scanning");

  var subscription2 =
      respeck!.connectionState.listen((BluetoothConnectionState state) async {
    if (state == BluetoothConnectionState.disconnected) {
      print(
          "${respeck?.disconnectReason?.code} ${respeck?.disconnectReason?.description}");
    }
  });
  respeck!.cancelWhenDisconnected(subscription2, delayed: true, next: true);

  if (respeck == null) {
    return;
  }

  await respeck!.connect();
  print("CONNECTED to Respeck!");
  showToast("Connected to ${respeck.toString()}");

  BluetoothService? ser;
  BluetoothCharacteristic? cha;

  List<BluetoothService> services = await respeck!.discoverServices();
  for (BluetoothService s in services) {
    if (s.serviceUuid.str == "00001523-1212-efde-1523-785feabcd125") {
      ser = s;
      break;
    }
  }

  if (ser != null) {
    var characteristics = ser.characteristics;
    for (BluetoothCharacteristic c in characteristics) {
      if (c.characteristicUuid.str == "00001524-1212-efde-1523-785feabcd125") {
        print("found accel characteristic");
        cha = c;
        break;
      }
    }
  }

  if (cha != null) {
    final subscription3 = cha.onValueReceived.listen((value) async {
      ui.received_packet = true;

      DateTime packet_received_ts = DateTime.now();

      Uint8List ul = Uint8List.fromList(value);
      ByteData bd = ul.buffer.asByteData();

      int ts = bd.getUint32(0);
      ts = (ts * 197 * 1000 / 32768).toInt();
      print(ts);

      int packetSeqNumber = bd.getUint16(4);
      print(packetSeqNumber);

      int? battLevel;
      bool charging = false;

      if (respeckVersion == 6) {
        battLevel = bd.getUint8(6);
        if (bd.getUint8(7) == 1) {
          charging = true;
        }
      }

      String csv_str = "";
      int seqNumInPacket = 0;
      double x = 0, y = 0, z = 0;

      for (int i = 8; i < bd.lengthInBytes; i += 6) {
        int b1 = bd.getInt8(i);
        int b2 = bd.getInt8(i + 1);
        x = combineAccelBytes(b1, b2);
        int b3 = bd.getInt8(i + 2);
        int b4 = bd.getInt8(i + 3);
        y = combineAccelBytes(b3, b4);
        int b5 = bd.getInt8(i + 4);
        int b6 = bd.getInt8(i + 5);
        z = combineAccelBytes(b5, b6);

        HARService.addSample(x, y, z);

        Task1Service.addSample(x, y, z);

        try {
          Task2Service.addSample(x, y, z);
        } catch (_) {
          // ignore
        }

        csv_str +=
            "${packet_received_ts.millisecondsSinceEpoch},$ts,$packetSeqNumber,$seqNumInPacket,$x,$y,$z\n";

        seqNumInPacket++;
        ui.recorded_samples++;
      }

      String? prediction;
      if (HARService.hasEnoughSamples()) {
        prediction = HARService.predict();
      }

      ui.updateUI(
        x: x,
        y: y,
        z: z,
        batteryLevel: battLevel,
        isCharging: charging,
        respeckVersion: respeckVersion,
        prediction: prediction,
        bufferSize: HARService.getBufferSize(),
      );

      if (ui.recording) {
        int elapsed_secs =
            packet_received_ts.difference(ui.start_timestamp!).inSeconds;

        ui.recording_info =
            "Written ${ui.recorded_samples} samples (${elapsed_secs} seconds)";

        await csvFile.writeAsString(csv_str, mode: FileMode.append, flush: false);
      }
    });

    respeck!.cancelWhenDisconnected(subscription3);

    await Future.delayed(const Duration(seconds: 3));
    await cha.setNotifyValue(true);
  }
}