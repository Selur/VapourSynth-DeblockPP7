project('DeblockPP7', 'cpp',
  default_options : ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++14'],
  meson_version : '>=0.51.0',
  version : '5'
)

sources = [
  'DeblockPP7/DeblockPP7.cpp',
  'DeblockPP7/DeblockPP7.hpp'
]

libs = []

# Prefer system VapourSynth headers, fall back to the vendored API 4 headers.
system_vs_dep = dependency('vapoursynth', version : '>=55', required : false)
if system_vs_dep.found()
  vapoursynth_dep = system_vs_dep.partial_dependency(compile_args : true, includes : true)
else
  vapoursynth_dep = declare_dependency(include_directories : include_directories('include/vapoursynth'))
endif

cpp = meson.get_compiler('cpp')
gcc_syntax = cpp.get_argument_syntax() == 'gcc'

if gcc_syntax
  add_project_arguments('-ffast-math', language : 'cpp')
else
  add_project_arguments('/fp:fast', language : 'cpp')
endif

if host_machine.cpu_family().startswith('x86')
  # SSE2/SSE4.1 code paths via vectorclass, selected at runtime (or with the opt parameter).
  sources += [
    'DeblockPP7/DeblockPP7_SSE2.cpp',
    'DeblockPP7/vectorclass/instrset.h',
    'DeblockPP7/vectorclass/instrset_detect.cpp',
    'DeblockPP7/vectorclass/vectorclass.h',
    'DeblockPP7/vectorclass/vectorf128.h',
    'DeblockPP7/vectorclass/vectori128.h'
  ]

  add_project_arguments('-DVS_TARGET_CPU_X86', language : 'cpp')
  if gcc_syntax
    add_project_arguments('-mfpmath=sse', '-msse2', language : 'cpp')
  endif

  # MSVC needs no flag for SSE4.1 intrinsics, the source defines __SSE4_1__ for vectorclass itself.
  libs += static_library('sse4', 'DeblockPP7/DeblockPP7_SSE4.cpp',
    dependencies : vapoursynth_dep,
    cpp_args : gcc_syntax ? ['-msse4.1'] : [],
    gnu_symbol_visibility : 'hidden'
  )
endif

# VapourSynth only autoloads *.dylib on macOS, while Meson defaults to *.so for modules.
suffix = host_machine.system() == 'darwin' ? 'dylib' : []

if get_option('python_package')
  # Same layout as the VapourSynth PyPI package autoloads from: site-packages/vapoursynth/plugins
  py = import('python').find_installation(pure : false)
  install_dir = join_paths(py.get_install_dir(), 'vapoursynth', 'plugins')
else
  libdir = join_paths(get_option('prefix'), get_option('libdir'))
  if system_vs_dep.found()
    libdir = system_vs_dep.get_variable(pkgconfig : 'libdir', default_value : libdir)
  endif
  install_dir = join_paths(libdir, 'vapoursynth')
endif

shared_module('deblockpp7', sources,
  dependencies : vapoursynth_dep,
  link_with : libs,
  install : true,
  install_dir : install_dir,
  name_suffix : suffix,
  gnu_symbol_visibility : 'hidden'
)
