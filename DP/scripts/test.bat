@echo off
set NUM_TERMINALS=3

for /L %%i in (1,1,%NUM_TERMINALS%) do (
    start "" cmd /k "%WINDIR%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command "& 'C:\Users\hperu\anaconda3\shell\condabin\conda-hook.ps1' ; conda activate 'C:\Users\hperu\anaconda3' " & exit"
)
