@echo off
setlocal EnableDelayedExpansion

REM Get target folder argument (relative to script folder)
set "targetFolder=%~dp0%1"

REM Normalize path (remove trailing backslash)
if "%targetFolder:~-1%"=="\" set "targetFolder=%targetFolder:~0,-1%"

REM File list
set "fileList=%~dp0files-to-delete.txt"

REM Check if the folder exists
if not exist "%targetFolder%" (
    echo ERROR: Folder "%targetFolder%" does not exist.
    exit /b 1
)

REM Check if the file list exists
if not exist "%fileList%" (
    echo ERROR: File list "%fileList%" not found.
    exit /b 1
)

echo Deleting files/folders listed in: %fileList%
echo Target folder: %targetFolder%
echo.

for /f "usebackq delims=" %%f in ("%fileList%") do (
    set "raw=%%f"

    REM Remove surrounding quotes
    set "line=!raw:"=!"

    REM Replace {target-Folder} with actual path
    set "resolvedLine=!line:{target-Folder}=%targetFolder%!"

    REM Convert forward slashes to backslashes
    set "resolvedLine=!resolvedLine:/=\!"

    REM Expand wildcards manually
    for /f "delims=" %%A in ('dir /b /s /a-d "!resolvedLine!" 2^>nul') do (
        del /f /q "%%A"
        if exist "%%A" (
            echo Failed to delete: %%A
        ) else (
            echo Deleted file: %%A
        )
    )

    REM Delete empty folders matching pattern
    for /f "delims=" %%D in ('dir /b /s /ad "!resolvedLine!" 2^>nul') do (
        rmdir /s /q "%%D" 2>nul
        if exist "%%D" (
            echo Failed to delete folder: %%D
        ) else (
            echo Deleted folder: %%D
        )
    )
)

endlocal
