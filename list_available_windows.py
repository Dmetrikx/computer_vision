import pygetwindow as gw

# Get all windows
windows = gw.getAllWindows()

# Print the titles of all windows
for window in windows:
    print(window.title)
