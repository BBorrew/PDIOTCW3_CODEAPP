import 'package:shared_preferences/shared_preferences.dart';
import 'dart:io';

// Subject and Respeck IDs are stored in shared preferences
final SharedPreferencesAsync asyncPrefs = SharedPreferencesAsync();
String? respeckUUID;
String? subjectID;
String? fwString;

// The folder and CSV file where sensor data is written
late Directory? storageFolder;
late File csvFile;

// Read later from pubspec.yaml
String appVersionName = "0.0.0";
int appVersionCode = 0;
