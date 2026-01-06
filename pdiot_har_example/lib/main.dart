import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'globals.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'home.dart';

import 'task2_page.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  PackageInfo.fromPlatform().then((PackageInfo packageInfo) {
    appVersionName = packageInfo.version;
    appVersionCode = int.parse(packageInfo.buildNumber);
  });

  respeckUUID = await asyncPrefs.getString('rid');
  print("respeckUUID:${respeckUUID}");

  subjectID = await asyncPrefs.getString('sid');
  print("subjectID:${subjectID}");

  storageFolder = await getDownloadsDirectory();

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'pdiot',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.lightBlue),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: "PDIoT"),
      routes: {
        '/task2': (_) => const Task2Page(),
      },
    );
  }
}