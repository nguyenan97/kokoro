"""Local inference settings; configured before loading Kokoro."""
import os


def configure_runtime():
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', os.getenv('KOKORO_CPU_THREADS', '2'))
    os.environ.setdefault('MKL_NUM_THREADS', os.getenv('KOKORO_CPU_THREADS', '2'))

    # Bundle eSpeak in the virtual environment rather than installing an MSI.
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    os.environ.setdefault('ESPEAK_DATA_PATH', str(espeakng_loader.get_data_path()))

    import torch
    torch.set_num_threads(int(os.getenv('KOKORO_CPU_THREADS', '2')))
    torch.set_num_interop_threads(1)
    if os.getenv('KOKORO_REQUIRE_CUDA', '0') == '1' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. Install the CUDA PyTorch wheel in .venv.')
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = os.getenv('KOKORO_CUDNN_BENCHMARK', '1') == '1'
    if os.name == 'nt':
        import ctypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x8000):
            raise ctypes.WinError(ctypes.get_last_error())
