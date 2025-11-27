import socket
import time


class UnixSocketClient:
    """
    A robust client for communicating over a Unix Domain Socket (UDS).
    Handles connection retries and line-based I/O.
    """

    def __init__(self, socket_path, max_retries=5, retry_delay=2.0):
        self.socket_path = socket_path
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.sock = None
        self.file_handle = None

    def connect(self):
        """
        Attempts to connect to the Unix Domain Socket.
        Retries up to self.max_retries times with a delay between attempts.
        """
        attempt = 1
        while attempt <= self.max_retries:
            try:
                print(f"Connecting to {self.socket_path} (Attempt {attempt}/{self.max_retries})...")

                # Create a Unix Domain Socket
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(self.socket_path)

                # Create a file-like object for easy line reading/writing
                # 'rw' = read/write, buffering=1 means line-buffered
                self.file_handle = self.sock.makefile('rw', buffering=1, encoding='utf-8')

                print("Successfully connected.")
                return True

            except (FileNotFoundError, ConnectionRefusedError, socket.error) as e:
                print(f"Connection failed: {e}")
                self._close_resources()

                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                attempt += 1

        print("Max retries reached. Could not connect.")
        return False

    def send_line(self, message):
        """
        Writes a line of text to the socket. Appends a newline if missing.
        Returns True if successful, False if connection was lost.
        """
        if not self.file_handle:
            if not self.connect():
                return False

        try:
            if not message.endswith('\n'):
                message += '\n'

            self.file_handle.write(message)
            self.file_handle.flush()  # Ensure data is pushed to the socket immediately
            return True

        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"Error sending data: {e}")
            self._close_resources()
            return False

    def read_line(self):
        """
        Reads a single line of text from the socket.
        Returns the string (stripped of newline) or None if error/EOF.
        """
        if not self.file_handle:
            if not self.connect():
                return None

        try:
            # readline will block until a newline is received or connection closes
            line = self.file_handle.readline()

            # Empty string indicates EOF (server closed connection)
            if not line:
                print("Server closed the connection.")
                self._close_resources()
                return None

            return line.rstrip('\n').rstrip('\r')

        except (ConnectionResetError, OSError, socket.timeout) as e:
            print(f"Error reading data: {e}")
            self._close_resources()
            return None

    def _close_resources(self):
        """Clean helper to close file handle and socket."""
        if self.file_handle:
            try:
                self.file_handle.close()
            except OSError:
                pass
            self.file_handle = None

        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def close(self):
        """Public method to close the connection manually."""
        self._close_resources()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
