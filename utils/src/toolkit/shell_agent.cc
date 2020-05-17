#include "toolkit/shell_agent.h"

#include <sys/wait.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <mutex>
#include <string>
#include "absl/strings/str_format.h"
#include "toolkit/string_agent.h"

using std::string;
using std::mutex;
using std::lock_guard;
using ps::toolkit::LineFileReader;

namespace ps {
namespace toolkit {

static pid_t *childpid = NULL;  /* ptr to array allocated at run-time */
static int   maxfd = 0;  /* from our open_max(), {Prog openmax} */
static long  openmax = 0;
/*
 * If OPEN_MAX is indeterminate, we're not
 * guaranteed that this is adequate.
 */
#define OPEN_MAX_GUESS 1024
static long open_max(void)  {
  if (openmax == 0) {  /* first time through */
    errno = 0;
    if ((openmax = sysconf(_SC_OPEN_MAX)) < 0) {
      if (errno == 0) {
        openmax = OPEN_MAX_GUESS; /* it's indeterminate */
      } else {
        printf("sysconf error for _SC_OPEN_MAX");
      }
    }
  }

  return(openmax);
}
#undef OPEN_MAX_GUESS

#define SHELL "/bin/sh"
static FILE *vpopen(const char *cmdstring, const char *type) {
  int   i, pfd[2];
  pid_t pid;
  FILE  *fp;

  /* only allow "r" or "w" */
  if ((type[0] != 'r' && type[0] != 'w') || type[1] != 0) {
    errno = EINVAL;     /* required by POSIX.2 */
    return NULL;
  }

  if (childpid == NULL) { /* first time through */
    /* allocate zeroed out array for child pids */
    maxfd = open_max();
    if ((childpid = (pid_t *)calloc(maxfd, sizeof(pid_t))) == NULL) {
      return NULL;
    }
  }

  if (pipe(pfd) < 0) {
    return NULL;   /* errno set by pipe() */
  }

  if ((pid = vfork()) < 0) {
    return NULL;   /* errno set by fork() */
  } else if (pid == 0) {   /* child */
    if (*type == 'r') {
      close(pfd[0]);
      if (pfd[1] != STDOUT_FILENO) {
        dup2(pfd[1], STDOUT_FILENO);
        close(pfd[1]);
      }
    } else {
      close(pfd[1]);
      if (pfd[0] != STDIN_FILENO) {
        dup2(pfd[0], STDIN_FILENO);
        close(pfd[0]);
      }
    }
    /* close all descriptors in childpid[] */
    for (i = 0; i < maxfd; i++) {
      if (childpid[i] > 0) {
        close(i);
      }
    }

    execl(SHELL, "sh", "-c", cmdstring, (char *) 0);
    _exit(127);
  }

  /* parent */
  if (*type == 'r') {
    close(pfd[1]);
    if ((fp = fdopen(pfd[0], type)) == NULL) {
      return NULL;
    }
  } else {
    close(pfd[0]);
    if ((fp = fdopen(pfd[1], type)) == NULL) {
      return NULL;
    }
  }

  childpid[fileno(fp)] = pid; /* remember child pid for this fd */
  return(fp);
}
#undef SHELL

static int vpclose(FILE *fp) {
  int   fd, stat;
  pid_t pid;

  if (childpid == NULL) {
    return -1;     /* popen() has never been called */
  }

  fd = fileno(fp);
  if ((pid = childpid[fd]) == 0) {
    return -1;     /* fp wasn't opened by popen() */
  }

  childpid[fd] = 0;
  if (fclose(fp) == EOF) {
    return -1;
  }

  while (waitpid(pid, &stat, 0) < 0) {
    if (errno != EINTR) {
      return -1; /* error other than EINTR from waitpid() */
    }
  }

  return stat;   /* return child's termination status */
}

static bool shell_verbose_ = true;
static mutex pipe_mtx_;

// popen and pclose are not thread-safe
static FILE *guarded_popen(const char* command, const char* type) {
  FILE *fd = NULL;
  {
    lock_guard<mutex> lock(pipe_mtx_);
    fd = vpopen(command, type);
    // fd = popen(command, type);
  }
  return fd;
}

static int guarded_pclose(FILE* stream) {
  int ret = 0;
  {
    lock_guard<mutex> lock(pipe_mtx_);
    ret = vpclose(stream);
    // ret = pclose(stream);
  }
  return ret;
}

void ShellAgent::set_shell_verbose(bool x) {
  shell_verbose_ = x;
}

shared_ptr<FILE> ShellAgent::shell_open_file(const string& path, const string& mode, const size_t buffer_size) {
  if (shell_verbose_) {
    LOG(INFO) << "Opening file [" << path << "] with mode [" << mode << "]";
  }

  FILE *fp = NULL;
  PCHECK(fp = fopen(path.c_str(), mode.c_str())) << "fopen() fail, path = [" << path << "], mode = [" << mode << "].";

  char *buffer = NULL;
  if (buffer_size > 0) {
    buffer = new char[buffer_size];
    PCHECK(0 == setvbuf(fp, buffer, _IOFBF, buffer_size));
  }

  return shared_ptr<FILE> (fp, [path, buffer](FILE * fp) {
    if (shell_verbose_) {
      LOG(INFO) << "Closing file [" << path << "]";
    }

    PCHECK(0 == fclose(fp)) << "fcolse() fail, path = [" << path << "].";
    delete[] buffer;
  });
}

shared_ptr<FILE> ShellAgent::shell_open_pipe(const string& cmd, const string& mode, const size_t buffer_size) {
  if (shell_verbose_) {
    LOG(INFO) << "Opening pipe [" << cmd << "] with mode [" << mode << "]";
  }

  FILE *fp = NULL;
  PCHECK(fp = guarded_popen(absl::StrFormat("set -o pipefail; %s", cmd.c_str()).c_str(), mode.c_str()))
    << "guarded_popen() fail, cmd = [" << cmd << "], mode = [" << mode << "]";

  char *buffer = NULL;
  if (buffer_size > 0) {
    buffer = new char[buffer_size];
    CHECK(0 == setvbuf(fp, buffer, _IOFBF, buffer_size));
  }

  return shared_ptr<FILE>(fp, [cmd, buffer](FILE *fp) {
    if (shell_verbose_) {
      LOG(INFO) << "Closing pipe [" << cmd << "]";
    }

    PCHECK(0 == guarded_pclose(fp)) << "guarded_pclose() fail, cmd = [" << cmd << ".";
    delete[] buffer;
  });
}

void ShellAgent::shell_execute(const string& cmd) {
  shell_open_pipe(cmd, "w");
}

string ShellAgent::shell_get_command_output(const string& cmd) {
  shared_ptr<FILE> pipe = shell_open_pipe(cmd, "r");
  LineFileReader reader;
  if (reader.getdelim(&*pipe, 0)) {
    return reader.get();
  }
  return "";
}

} // namespace toolkit
} // namespace ps

