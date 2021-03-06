pipeline {
    agent none
    triggers {
        // cron('H H(4-5) * * *')
        pollSCM('H/10 * * * *')
    }
    stages {
        stage('main') {
            matrix {
                axes {
                    // axis {
                    //     name 'build'
                    //     values 'cmake'
                    // }
                    axis {
                        name 'host'
                        values 'Almaha', 'Buraq', 'Condor', 'Flamingo',
                            'Jasmine', 'Shihab', 'Vulture', 'Albatross',
                            'Tuwaiq'
                            // 'stork'   // no modules
                            // 'Oqab',   // decommissioned
                            // 'P100',   // decommissioned
                            // 'Raed',   // decommissioned
                            // 'Thana',  // decommissioned
                            // 'Bashiq', // decommissioned
                    }
                } // axes
                stages {
                   stage('build and run') {
                        // agent {node "${host}.kaust.edu.sa"}
                        agent {node "${host}"}
                        steps {
                            sh '''#!/bin/bash -le
                            hostname && pwd

                            # modules
                            echo "========================================"

                            module purge

                            ####################################################
                            # gcc and cmake
                            ####################################################

                            if [ "${host}" = "Vulture" ]; then
                                module load gcc/7.2.0
                                module load cmake/3.17.3
                            else
                                module load gcc/10.2.0
                                module load cmake/3.19.2
                            fi

                            ####################################################
                            # BLAS/LAPACK
                            ####################################################

                            module load mkl/2020.0.166

                            module list

                            echo "========================================"

                            rm -rf build
                            mkdir build
                            cd build
                            cmake -Dcolor=no -DCMAKE_CXX_FLAGS="-Werror" -Dlog=trace ..
                            export top=../..

                            echo "========================================"
                            make -j8

                            echo "========================================"
                            ldd test/tester

                            echo "========================================"
                            cd test
                            ./run_tests.py --small --xml ${top}/report.xml
                            '''
                        } // steps
                        post {
                            failure {
                                mail to: 'mohammed.farhan@kaust.edu.sa',
                                    subject: "${currentBuild.fullDisplayName} -- ${host} failed",
                                    body: "see more at ${env.BUILD_URL}"
                            }
                            always {
                                junit '*.xml'
                            }
                        } // post
                    } // stage('build and run')
                } // stages
            } // matrix
        } // stage('main')
    } // stages
} // pipeline
