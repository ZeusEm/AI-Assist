@echo off
rem This command turns off the command echoing feature. 
rem When echo is off, the commands in the batch file are not displayed on the command prompt window while they are being executed. 
rem This results in a cleaner output.

"C:\Users\Public\anaconda3\python.exe" "ai_assist.py" %*
rem This command runs the ai_assist.py Python script using the Anaconda Python interpreter located at "C:\Users\Public\anaconda3\python.exe".
rem - "C:\Users\Public\anaconda3\python.exe": Specifies the path to the Python executable.
rem - "ai_assist.py": Specifies the Python script to be executed.
rem - %*: This variable represents all the arguments passed to the batch script. 
rem        It allows any additional command line arguments to be forwarded to the Python script.

pause
rem This command pauses the execution of the batch file and waits for the user to press any key before continuing.
rem This is useful for keeping the command prompt window open after the script finishes executing, so the user can see any output or messages.
