import 'package:flutter/material.dart';
import 'globals.dart';
import 'scanning.dart';

// The settings page shows the app version and allows the app to be paired
//// with a subject and respeck.

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  String subject_id = "";
  String respeck_uuid = "";
  TextEditingController textController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: <Widget>[
            const Text(
              'App version',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            Text(
              appVersionName,
            ),
            const SizedBox(height: 10),
            const Text(
              'Current pairing',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            Text(
              'Subject=$subjectID, Respeck=$respeckUUID',
            ),
            const SizedBox(height: 10),
            const Text(
              'Storage folder',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20.0),
              child: Text(
                storageFolder!.path,
              ),
            ),
            const SizedBox(height: 10),
            const Text(
              'New pairing',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            TextField(
              decoration: const InputDecoration(hintText: 'Subject ID'),
              onChanged: (text) {
                subject_id = text;
              },
            ),
            TextField(
              controller: textController,
              decoration: const InputDecoration(hintText: 'Respeck UUID'),
              onChanged: (text) {
                respeck_uuid = text;
              },
            ),
            const SizedBox(height: 10),
            Row(
              mainAxisAlignment:
                  MainAxisAlignment.spaceEvenly, // Align buttons evenly
              children: [
                ElevatedButton(
                    onPressed: () {
                      _scanQRCode(context);
                    },
                    child: const Text('Scan QR code')),
                const SizedBox(height: 10),
                ElevatedButton(
                  onPressed: () async {
                    // Store the new subject and respeck IDs in shared preferences
                    // to survive app restarts
                    await asyncPrefs.setString('rid', respeck_uuid);
                    await asyncPrefs.setString('sid', subject_id);
                    setState(() {
                      respeckUUID = respeck_uuid;
                      subjectID = subject_id;
                    });
                  },
                  child: const Text('Save settings'),
                ),
              ],
            )
          ],
        ),
      ),
    );
  }

  // Launch the QR code scanner, which appears in a new page
  Future<void> _scanQRCode(BuildContext context) async {
    final result = await Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const ScanningPage()),
    );

    // When a BuildContext is used from a StatefulWidget, the mounted property
    // must be checked after an asynchronous gap.
    if (!context.mounted) return;

    respeck_uuid = result;
    textController.text = result;
  }
}
