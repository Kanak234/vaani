[Setup]
AppName=Vaani
AppVersion=1.0.1
AppPublisher=Vaani Project
DefaultDirName={autopf}\Vaani
DefaultGroupName=Vaani
OutputDir=..\..\release\windows
OutputBaseFilename=vaani_installer
SetupIconFile=..\vaani.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
LicenseFile=..\..\LICENSE

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\..\release\windows\vaani\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Vaani"; Filename: "{app}\vaani.exe"; IconFilename: "{app}\packaging\vaani.ico"
Name: "{autodesktop}\Vaani"; Filename: "{app}\vaani.exe"; IconFilename: "{app}\packaging\vaani.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\vaani.exe"; Description: "{cm:LaunchProgram,Vaani}"; Flags: nowait postinstall skipifsilent
