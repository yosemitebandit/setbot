"""Deploying the site.

e.g.
  $ fab kepler deploy
"""

import os

from fabric.api import env, run, local, sudo, put, cd


def kepler():
  """The linode host."""
  env.use_ssh_config = True
  env.user = 'matt'
  env.hosts = ['kepler']
  env.remote_project_dir = '/home/matt/setbot.oakmachine.com'
  env.branch = 'master'


def deploy():
  """Updates the site."""
  # Zips up the contents of setbot-server/ and sends to the remote.
  tgz_file = 'setbot-server-files.tgz'
  local('tar -czvf %s setbot-server' % tgz_file)
  put(tgz_file, '/tmp/')
  # Removes the local copy and the old project dir on the remote.
  local('rm %s' % tgz_file)
  run('rm -rf %s' % os.path.join(env.remote_project_dir, 'setbot-server'))
  run('mv /tmp/%s %s' % (tgz_file, env.remote_project_dir))
  with cd(env.remote_project_dir):
    run('tar -xf %s' % os.path.join(env.remote_project_dir, tgz_file))
    run('rm %s' % tgz_file)
  # Restarts via upstart.
  sudo('restart setbot-server')
