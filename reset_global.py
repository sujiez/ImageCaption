import configuration as conf


fd2 = open(conf.global_step_file, 'w')
fd2.write(str(0))
fd2.close()

