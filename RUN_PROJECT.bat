@echo off
title Week 2: Supervised Machine Learning Pipeline
cd /d "%~dp0"
echo ====================================================================
echo Starting Supervised Machine Learning Models Pipeline...
echo ====================================================================
echo.
python main.py
echo.
echo ====================================================================
echo Pipeline Execution Completed!
echo Generating Word Document Report...
echo ====================================================================
python generate_report.py
echo.
echo Opening Project Folder and Report...
start "" "%~dp0Week_2_Supervised_Machine_Learning_Report.docx"
start "" "%~dp0visualizations"
echo.
echo Press any key to exit this window.
pause > nul
