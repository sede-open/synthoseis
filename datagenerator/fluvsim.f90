!-----------------------------------------------------------------------
!
!              Hierarchical Fluvial Reservoir Modeling
!              ***************************************
!
! The program is executed with no command line arguments.  The user
! will be prompted for the name of a parameter file.  The parameter
! file is described in the documentation (see the example fluvsim.par)
!
! The output file will be a GEOEAS file containing the simulated facies
! codes.  The file is ordered by x,y,z, and then simulation (i.e., x
! cycles fastest, then y, then z, then realization number).
!
! Although somewhat odd, the facies coding is as follows (for input well
! conditioning...):
!
!                    0 = floodplain shale
!                    1 = channel sand (2 reserved for channel margin)
!                    3 = levee sand
!                    4 = crevasse sand
!
!
! AUTHOR: Clayton V. Deutsch          DATE: Original 1996 - Revised 2000
!-----------------------------------------------------------------------


!
! Module to declare dynamic arrays in multiple subroutines:
!
module geostat

    real, allocatable :: pmap(:, :, :), pmapa(:, :, :), pcurve(:, :), &
            pcurvea(:, :), flwz(:, :), creprz(:, :), cx(:, :, :), cz(:), cw(:, :), &
            ct(:, :, :), ccent(:, :), cprob(:), match(:), missm(:), cang(:), &
            ccosa(:), csina(:), ctana(:), cxorigin(:), cyorigin(:), chx(:), &
            rarray(:), crt(:, :), shiftp(:), atcut(:), atcdf(:)

    integer, allocatable :: xw(:), yw(:), zw(:), fw(:), ninwint(:), &
            facint(:), ixint(:, :), iyint(:, :), izint(:, :), nyc(:), ncre(:), &
            crxloc(:, :), cryloc(:, :), crnum(:, :), icon(:), icoff(:), &
            shiftc(:)

    integer*2, allocatable :: chanind(:, :, :), channum(:, :, :), &
            tchanind(:, :, :), tchannum(:, :, :), ixlotem(:, :), ixhitem(:, :), &
            izlotem(:, :, :), izhitem(:), llwid(:, :), rlwid(:, :), llzlo(:, :), &
            rlzlo(:, :), llbh(:, :, :), rlbh(:, :, :)

    logical, allocatable :: chanon(:), cre(:, :, :), crright(:, :)

    real VERSION, EPSLON, DEGTOR, fco(3), fcad(3), fcal(3), fct(3), fctu(3), &
            fctul(3), fcwt(3), fcwu(3), fcwul(3), tarprop(0:4), modprop(0:4), &
            flw(3), flh(3), fld(3), fcrlen(3), fcrt(3), fcrnw(3), fcrwl(3), &
            fcrlat(3), creprob(3), cumprob(3), xsiz, ysiz, zsiz, xmn, xmx, ymn, &
            ymx, zmn, wellmis, welltry, avgthick, sclglob, sclvert, sclarea, &
            sclwell, wfac, objmin, t0, redfac, cxmin, cxmax, cymin, cymax, cdy

    integer nx, ny, nz, nsim, nwd, nwint, nc, MAXW, MAXDAT, MAXC, MAXL, MAXBW, &
            MAXCRE, MAXCRX, MAXCRY, MAXCPERT, MAXDIS, lin, ldbg, lgeo, &
            lout, lpv, lpa, lwd, ichan, ilev, icre, idbg, mxc, niter, mnoc, &
            kasas, ksas, numred, nonp, noffp, nvshift, nfct2

    logical lglob, lvert, lvertng, larea, lareang, lwell

end module


program main
    !-----------------------------------------------------------------------
    !
    !              Hierarchical Fluvial Reservoir Modeling
    !              ***************************************
    !
    ! Read the parameters, loop over each realization to generate, close
    ! output files and stop.
    !
    !-----------------------------------------------------------------------
    use       geostat
    !
    ! Some basic initialization:
    !
    VERSION = 2.900
    EPSLON = 1.0e-05
    DEGTOR = 3.14159 / 180.0
    !
    ! Read the parameters and data:
    !
    call readparm
    !
    ! Call fluvsim for the simulation:
    !
    do isim = 1, nsim
        write(*, *)
        write(*, *) 'Working on realization number ', isim
        call fluvsim
    end do
    close(ldbg)
    close(lout)
    close(lgeo)
    close(lpv)
    close(lpa)
    close(lwd)
    !
    ! Finished:
    !
    write(*, *)
    write(*, 9998) VERSION
    9998 format(/' FLUVSIM Version: ', f5.3, ' Finished'/)
    stop
end


subroutine readparm
    !-----------------------------------------------------------------------
    !
    !                  Initialization and Read Parameters
    !                  **********************************
    !
    ! The input parameters and data are read in from their files. Some quick
    ! error checking is performed and the statistics of all the variables
    ! being considered are written to standard output.
    !
    !
    !
    !-----------------------------------------------------------------------
    use       geostat
    !      use       msflib
    integer   icolv(3), icola(3), test
    real      var(100), proptest(3)
    real*8    p, acorni
    character datafl*132, dbgfl*132, geofl*132, outfl*132, pcurout*132, &
            pmapout*132, wellout*132, pcurvefl*132, pmapfl*132, str*132
    logical   testfl, inflag
    !
    ! ACORN parameters:
    !
    parameter(KORDEI = 12, MAXOP1 = KORDEI + 1, MAXINT = 2**30)
    common /iaco/ ixv(MAXOP1)
    !
    ! Input/Output units used:
    !
    lin = 1
    ldbg = 2
    lgeo = 3
    lout = 4
    lpv = 7
    lpa = 8
    lwd = 9
    !
    ! Note VERSION number:
    !
    write(*, 9999) VERSION
    9999 format(/' FLUVSIM Version: ', f5.3/)
    !
    ! Get the name of the parameter file - try the default name if no input:
    !
    str(1:1) = ' '
    !      call getarg(1,str)
    !      if(str(1:1).eq.' ')then
    !            write(*,*) 'Which parameter file do you want to use?'
    !            read (*,'(a20)') str(1:20)
    !      end if
    if(str(1:1).eq.' ') str = 'fluvsim.par         '
    inquire(file = str, exist = testfl)
    if(.not.testfl) then
        write(*, *) 'ERROR - the parameter file does not exist,'
        write(*, *) '        check for the file and try again  '
        write(*, *)
        if(str(1:20).eq.'fluvsim.par         ') then
            write(*, *) '        creating a blank parameter file'
            call makepar
            write(*, *)
        end if
        stop
    endif
    open(lin, file = str, status = 'OLD')
    !
    ! Find Start of Parameters:
    !
    1    read(lin, '(a4)', end = 98) str(1:4)
    if(str(1:4).ne.'STAR') go to 1
    !
    ! Read Input Parameters:
    !
    read(lin, '(a40)', err = 98) datafl
    call chknam(datafl, 40)
    write(*, *) ' data file = ', datafl

    read(lin, *, err = 98) iwl, ixl, iyl, izl, ifl
    write(*, *) ' input columns = ', iwl, ixl, iyl, izl, ifl

    read(lin, *, err = 98) tmin, tmax
    write(*, *) ' trimming limits = ', tmin, tmax

    read(lin, *, err = 98) idbg
    write(*, *) ' debugging level = ', idbg

    read(lin, '(a40)', err = 98) dbgfl
    call chknam(dbgfl, 40)
    write(*, *) ' debugging file = ', dbgfl
    open(ldbg, file = dbgfl, status = 'UNKNOWN')

    read(lin, '(a40)', err = 98) geofl
    call chknam(geofl, 40)
    write(*, *) ' geometry file = ', geofl

    read(lin, '(a40)', err = 98) outfl
    call chknam(outfl, 40)
    write(*, *) ' output file = ', outfl

    read(lin, '(a40)', err = 98) pcurout
    call chknam(pcurout, 40)
    write(*, *) ' output file for vertical proportions= ', pcurout

    read(lin, '(a40)', err = 98) pmapout
    call chknam(pmapout, 40)
    write(*, *) ' output file for areal proportions= ', pmapout

    read(lin, '(a40)', err = 98) wellout
    call chknam(wellout, 40)
    write(*, *) ' output file for well data= ', wellout

    read(lin, *, err = 98) nsim
    write(*, *) ' number of realizations = ', nsim

    read(lin, *, err = 98) nx, xmn, xsiz
    write(*, *) ' X grid specification = ', nx, xmn, xsiz

    read(lin, *, err = 98) ny, ymn, ysiz
    write(*, *) ' Y grid specification = ', ny, ymn, ysiz

    read(lin, *, err = 98) nz, avgthick
    write(*, *) ' Z grid specification = ', nz, avgthick
    zsiz = 1.0 / real(nz)
    zmn = 0.5 * zsiz

    allocate (flwz(nz, 3), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop

    read(lin, *, err = 98) ixv(1)
    write(*, *) ' random number seed = ', ixv(1)
    do i = 1, 1000
        p = acorni(idum)
    end do

    read(lin, *, err = 98) i1, i2, i3, i4
    write(*, *) ' Components in objective func = ', i1, i2, i3, i4
    lglob = .false.
    lvert = .false.
    larea = .false.
    lwell = .false.
    if(i1.ge.1) lglob = .true.
    if(i2.ge.1) lvert = .true.
    if(i3.ge.1) larea = .true.
    if(i4.ge.1) lwell = .true.

    read(lin, *, err = 98) sclglob, sclvert, sclarea, sclwell
    write(*, *) ' scaling of objective func = ', sclglob, sclvert, &
            sclarea, sclwell
    totwt = sclglob + sclvert + sclarea + sclwell
    if(totwt.le.0.0) stop 'Weights must be greater than zero'
    sclglob = sclglob / totwt
    sclvert = sclvert / totwt
    sclarea = sclarea / totwt
    sclwell = sclwell / totwt

    read(lin, *, err = 98) niter, mnoc, objmin
    write(*, *) ' Number of iterations, min obj = ', niter, mnoc, objmin

    read(lin, *, err = 98) t0, redfac, kasas, ksas, numred
    write(*, *) ' annealing schedule = ', t0, redfac, kasas, ksas, numred

    read(lin, *, err = 98) cumprob(1), cumprob(2), cumprob(3)
    write(*, *) ' perturbation probabilities = ', &
            cumprob(1), cumprob(2), cumprob(3)
    cumprob(2) = cumprob(1) + cumprob(2)
    cumprob(3) = cumprob(2) + cumprob(3)

    read(lin, *, err = 98) ichan, ilev, icre
    write(*, *) ' Facies types = ', ichan, ilev, icre

    read(lin, *, err = 98) tarprop(1), tarprop(3), tarprop(4)
    if(ilev.eq.0) tarprop(3) = 0.0
    if(icre.eq.0) tarprop(4) = 0.0
    write(*, *) ' Reference proportions = ', (tarprop(i), i = 1, 4)
    tarprop(0) = 1.0 - (tarprop(1) + tarprop(3) + tarprop(4))
    tarprop(2) = 0.0

    read(lin, '(a40)', err = 98) pcurvefl
    call chknam(pcurvefl, 40)
    write(*, *) ' vertical proportion curve file = ', pcurvefl

    read(lin, *, err = 98) itest
    write(*, *) ' net-to-gross or all facies = ', itest
    lvertng = .true.
    if(itest.eq.1) lvertng = .false.

    read(lin, *, err = 98) (icolv(i), i = 1, 3)
    write(*, *) ' column numbers = ', (icolv(i), i = 1, 3)

    read(lin, '(a40)', err = 98) pmapfl
    call chknam(pmapfl, 40)
    write(*, *) ' areal proportion map file = ', pmapfl

    read(lin, *, err = 98) itest
    write(*, *) ' net-to-gross or all facies = ', itest
    lareang = .true.
    if(itest.eq.1) lareang = .false.

    read(lin, *, err = 98) (icola(i), i = 1, 3)
    write(*, *) ' column numbers = ', (icola(i), i = 1, 3)

    read(lin, *, err = 98) mxc
    write(*, *) ' maximum number of channels = ', mxc

    read(lin, *, err = 98) (fco(i), i = 1, 3)
    write(ldbg, *) ' channel orientation = ', (fco(i), i = 1, 3)

    read(lin, *, err = 98) (fcad(i), i = 1, 3)
    write(ldbg, *) ' channel sinuosity dep = ', (fcad(i), i = 1, 3)

    read(lin, *, err = 98) (fcal(i), i = 1, 3)
    write(ldbg, *) ' channel sinuosity len = ', (fcal(i), i = 1, 3)

    read(lin, *, err = 98) (fct(i), i = 1, 3)
    write(ldbg, *) ' channel thickness = ', (fct(i), i = 1, 3)

    read(lin, *, err = 98) (fctu(i), i = 1, 3)
    write(ldbg, *) ' channel thickness undulation = ', (fctu(i), i = 1, 3)

    read(lin, *, err = 98) (fctul(i), i = 1, 3)
    write(ldbg, *) ' channel thickness undul len=', (fctul(i), i = 1, 3)

    read(lin, *, err = 98) (fcwt(i), i = 1, 3)
    write(ldbg, *) ' channel W/T ratio = ', (fcwt(i), i = 1, 3)

    read(lin, *, err = 98) (fcwu(i), i = 1, 3)
    write(ldbg, *) ' channel width undulation = ', (fcwu(i), i = 1, 3)

    read(lin, *, err = 98) (fcwul(i), i = 1, 3)
    write(ldbg, *) ' channel width undulation len = ', (fcwul(i), i = 1, 3)

    read(lin, *, err = 98) (flw(i), i = 1, 3)
    write(ldbg, *) ' levee width = ', (flw(i), i = 1, 3)
    do iz = 1, nz
        flwz(iz, 1) = flw(1)
        flwz(iz, 2) = flw(2)
        flwz(iz, 3) = flw(3)
    end do

    read(lin, *, err = 98) (flh(i), i = 1, 3)
    write(ldbg, *) ' levee height = ', (flh(i), i = 1, 3)

    read(lin, *, err = 98) (fld(i), i = 1, 3)
    write(ldbg, *) ' levee depth below channel top = ', (fld(i), i = 1, 3)

    read(lin, *, err = 98) (fcrlen(i), i = 1, 3)
    write(ldbg, *) ' crevasse attachment length = ', (fcrlen(i), i = 1, 3)

    read(lin, *, err = 98) (fcrt(i), i = 1, 3)
    write(ldbg, *) ' crevasse thickness = ', (fcrt(i), i = 1, 3)

    read(lin, *, err = 98) (fcrwl(i), i = 1, 3)
    write(ldbg, *) ' crevasse areal dimension = ', (fcrwl(i), i = 1, 3)

    write(*, *)
    write(*, *) ' Finished reading parameters'
    close(lin)
    !
    ! Dynamic memory allocation:
    !

    !
    !   MAXCRE    maximum number of crevasse templates
    !   MAXCPERT  maximum number of channels turned on/off/revised during
    !             a perturbation iteration
    !   MAXCRX    maximum crevasse template size in X
    !   MAXCRY    maximum crevasse template size in Y
    !   MAXW      maximum 1/2 width of channel (depends on width)
    !   MAXBW     maximum levee bank width
    !

    MAXL = 400 * max(nx, ny)   ! Modified from 2 * max(nx, ny)
    MAXC = mxc + 1
    !      MAXCRE   = 50
    MAXCRE = int(nx / 3. + .5)
    MAXCPERT = 10
    MAXDIS = 50
    !      MAXCRX   = 25
    !      MAXCRY   = 15
    !      MAXW  = 75
    !      MAXBW = 10
    MAXCRX = int(nx / 2. + .5)
    MAXCRY = int(nx / 2. + .5)
    MAXW = int(nx / 3. + .5)
    MAXBW = 400 * max(nx, ny)  ! Modified from 2 * max(nx, ny)

    allocate (pmapa(nx, ny, 3), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (pmap(nx, ny, 3), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (pcurve(nz, 3), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (pcurvea(nz, 3), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (creprz(nz, 3), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop

    allocate (chanind(nx, ny, nz), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (channum(nx, ny, nz), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (tchanind(nx, ny, nz), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (tchannum(nx, ny, nz), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop

    allocate (cx(MAXC, MAXL, 7), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (cz(MAXC), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (cw(MAXC, MAXL), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (ct(MAXC, MAXL, 7), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (ccent(MAXC, MAXL), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (cprob(MAXC), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (match(MAXC), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (missm(MAXC), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (cang(MAXC), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (ccosa(MAXC), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (csina(MAXC), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (ctana(MAXC), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (cxorigin(MAXC), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (cyorigin(MAXC), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (chx(MAXL), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (rarray(MAXL), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (nyc(MAXC), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (ncre(MAXC), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop

    allocate (ixlotem(MAXC, MAXL), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (ixhitem(MAXC, MAXL), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (izlotem(MAXC, MAXL, -MAXW:MAXW), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (izhitem(MAXC), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (llwid(MAXC, MAXL), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (rlwid(MAXC, MAXL), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (llzlo(MAXC, MAXL), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (rlzlo(MAXC, MAXL), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (llbh(MAXC, MAXL, MAXBW), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (rlbh(MAXC, MAXL, MAXBW), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop

    allocate (crt(MAXC, MAXCRE), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (shiftp(MAXCPERT), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (atcut(MAXDIS), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (atcdf(MAXDIS), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (crxloc(MAXC, MAXCRE), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (cryloc(MAXC, MAXCRE), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (crnum(MAXC, MAXCRE), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (icon(MAXCPERT), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (icoff(MAXCPERT), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (shiftc(MAXCPERT), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (chanon(MAXC), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (cre(MAXCRE, 0:MAXCRX, -MAXCRY:MAXCRY), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (crright(MAXC, MAXCRE), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop

    MAXDAT = 0
    inquire(file = datafl, exist = testfl)
    if(testfl.and.lwell) then
        open(lin, file = datafl, status = 'OLD')
        read(lin, *, err = 99)
        read(lin, *, err = 99) nvari
        do i = 1, nvari
            read(lin, *, err = 99)
        end do
        52         continue
        read(lin, *, err = 53, end = 53) (var(j), j = 1, nvari)
        MAXDAT = MAXDAT + 1
        go to 52
        53         continue
    end if
    if(MAXDAT.le.0) MAXDAT = 1

    allocate (xw(MAXDAT), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (yw(MAXDAT), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (zw(MAXDAT), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (fw(MAXDAT), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (ninwint(MAXDAT), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (facint(MAXDAT), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (ixint(MAXDAT, MAXDAT), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (iyint(MAXDAT, MAXDAT), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    allocate (izint(MAXDAT, MAXDAT), stat = test)
    if (test.ne.0) write(*, *) 'Error: Allocation failed: ', test
    if (test.ne.0) stop
    !
    ! Initialization.  Rescale vertical coordinates to average thickness.
    ! Scale annealing parameters by maximum number of channels.  Get
    ! distribution of apparent thicknesses:
    !
    do i = 1, 3
        fct(i) = fct(i) / avgthick
        fcwt(i) = fcwt(i) * avgthick
        fcrt(i) = fcrt(i) / avgthick
    end do
    nfct2 = 1 + int(fct(2) * 0.5 / zsiz)
    kasas = kasas * mxc
    ksas = ksas * mxc
    write(*, *)
    write(*, *) ' calling getat'
    call getat
    write(*, *) ' finished with getat'
    !
    ! Make sure the angles are in the right direction:
    !
    if(fco(1).gt.135.0) fco(1) = fco(1) - 180.0
    if(fco(2).gt.135.0) fco(2) = fco(2) - 180.0
    if(fco(3).gt.135.0) fco(3) = fco(3) - 180.0
    if(fco(1).lt.-45.0) fco(1) = fco(1) + 180.0
    if(fco(2).lt.-45.0) fco(2) = fco(2) + 180.0
    if(fco(3).lt.-45.0) fco(3) = fco(3) + 180.0
    !
    ! The limits of interest for the channels and the spacing of the slices
    ! that make up the channels:
    !
    xmx = xmn + (real(nx) - 0.5) * xsiz
    ymx = ymn + (real(ny) - 0.5) * ysiz
    xadd = 0.5 * xsiz + min((0.15 * (xmx - xmn)), (0.7071 * fct(2) * fcwt(2)))
    write (*, *) xadd
    cxmin = xmn - (0.5 * xsiz + xadd)
    cxmax = xmx + xadd
    yadd = 0.5 * ysiz + min((0.15 * (ymx - ymn)), (0.7071 * fct(2) * fcwt(2)))
    write (*, *) yadd
    cymin = ymn - (0.5 * ysiz + yadd)
    cymax = ymx + yadd
    cdy = 0.7071 * min(xsiz, ysiz)
    ndymax = sqrt((cxmax - cxmin)**2 + (cymax - cymin)**2) / cdy
    write(ldbg, 120) (xmn - 0.5 * xsiz), xmx, (ymn - 0.5 * ysiz), ymx, &
            cxmin, cxmax, cymin, cymax, cdy, ndymax
    write(*, 120)    (xmn - 0.5 * xsiz), xmx, (ymn - 0.5 * ysiz), ymx, &
            cxmin, cxmax, cymin, cymax, cdy, ndymax
    120  format(/, 'Limits of model X: ', 2f10.2, &
            /, '                Y: ', 2f10.2, &
            /, '   outer limits X: ', 2f10.2, &
            /, '                Y: ', 2f10.2, &
            /, '       channel dy: ', f10.4, &
            /, '       maximum ny: ', i10)
    !
    ! Set up the crevasse templates and figure out the probability of
    ! one per channel:
    !
    if(icre.eq.1) then
        write(*, *)
        write(*, *) 'Establishing ', MAXCRE, ' crevasse templates'
        !
        ! - scale input to grid units (and hard code some parameters):
        !
        fcrlen(1) = int(fcrlen(1) / cdy + 0.5)
        fcrlen(2) = int(fcrlen(2) / cdy + 0.5)
        fcrlen(3) = int(fcrlen(3) / cdy + 0.5)
        fcrnw(1) = 5
        fcrnw(2) = 10
        fcrnw(3) = 15
        fcrlat(1) = 0.5
        fcrlat(2) = 0.5
        fcrlat(3) = 0.5
        fcrwl(1) = int(1.5 * fcrwl(1) / cdy + 0.5)
        fcrwl(2) = int(1.5 * fcrwl(2) / cdy + 0.5)
        fcrwl(3) = int(1.5 * fcrwl(3) / cdy + 0.5)
        !
        ! - establish the crevasse templates and the number of crevasses:
        !
        call getcre
        cresize = 0.0
        do ic = 1, MAXCRE
            do ix = 0, MAXCRX
                do iy = -MAXCRY, MAXCRY
                    if(cre(ic, ix, iy)) cresize = cresize + &
                            cdy * cdy * fcrt(2) * (MAXCRX - real(ix)) / MAXCRX
                end do
            end do
        end do
        cresize = cresize / MAXCRE
        chsize = 0.55 * fct(2) * fct(2) * fcwt(2) * real(ny) * cdy
        creprob(2) = tarprop(4) / tarprop(1) * (chsize / cresize)
        !
        ! Scale the number of crevasses so that the proportion works out.
        !
        creprob(2) = creprob(2) / 2.0

        creprob(1) = creprob(2) / 1.5
        creprob(3) = creprob(2) * 1.5
        write(*, *) 'Average channel  size: ', chsize
        write(*, *) 'Average crevasse size: ', cresize
        write(*, *) 'There will be about ', int(creprob(2) + 0.5), &
                ' crevasses per channel'
        do iz = 1, nz
            creprz(iz, 1) = creprob(1)
            creprz(iz, 2) = creprob(2)
            creprz(iz, 3) = creprob(3)
        end do
    end if
    !
    ! Read the well data (if the file exists):
    !
    nwd = 0
    nwint = 0
    inquire(file = datafl, exist = testfl)
    if(testfl.and.lwell) then
        write(*, *)
        write(*, *) 'Reading input well data'
        open(lin, file = datafl, status = 'OLD')
        read(lin, *, err = 99)
        read(lin, *, err = 99) nvari
        do i = 1, nvari
            read(lin, *, err = 99)
        end do
        if(ixl.gt.nvari.or.iyl.gt.nvari.or.izl.gt.nvari.or.&
                ifl.gt.nvari) then
            write(*, *) 'ERROR: you have asked for a column number'
            write(*, *) '       greater than number in well data'
            stop
        end if
        if(ixl.le.0.or.iyl.le.0.or.izl.le.0.or.ifl.le.0) then
            write(*, *) 'ERROR: you must have coordinates and'
            write(*, *) '       facies in well data'
            stop
        end if
        !
        ! Read all the data until the end of the file:
        !
        iwold = -1
        ifold = -1
        5          read(lin, *, end = 6, err = 99) (var(j), j = 1, nvari)
        if(var(ifl).lt. tmin.or.var(ifl).ge. tmax.or.&
                var(ixl).lt.cxmin.or.var(ixl).ge.cxmax.or.&
                var(iyl).lt.cymin.or.var(iyl).ge.cymax.or.&
                var(izl).lt.  0.0.or.var(izl).ge.  1.0)    go to 5
        nwd = nwd + 1
        if(nwd.gt.MAXDAT) then
            write(*, *) ' ERROR exceeded MAXDAT - check inc file'
            stop
        end if
        !
        ! Acceptable data, assign the value, X, Y, Z coordinates, and weight:
        !
        iwell = var(iwl)
        fw(nwd) = var(ifl)
        call getindx(nx, xmn, xsiz, var(ixl), ii, inflag)
        xw(nwd) = max(min(ii, nx), 1)
        call getindx(ny, ymn, ysiz, var(iyl), ii, inflag)
        yw(nwd) = max(min(ii, ny), 1)
        call getindx(nz, zmn, zsiz, var(izl), ii, inflag)
        zw(nwd) = max(min(ii, nz), 1)
        !
        ! Check and see if this is a duplicate well interval:
        !
        do iii = 1, nwd - 1
            if((abs(xw(nwd) - xw(iii)) + &
                    abs(yw(nwd) - yw(iii)) + &
                    abs(zw(nwd) - zw(iii))).lt.1.0) then
                nwd = nwd - 1
                go to 5
            end if
        end do
        !
        ! Keep track of well intervals:
        !
        if(iwell.ne.iwold.or.fw(nwd).ne.ifold) then
            nwint = nwint + 1
            iwold = iwell
            ifold = fw(nwd)
            ninwint(nwint) = 1
            facint(nwint) = ifold
            ixint(nwint, 1) = xw(nwd)
            iyint(nwint, 1) = yw(nwd)
            izint(nwint, 1) = zw(nwd)
        else
            ninwint(nwint) = ninwint(nwint) + 1
            ii = ninwint(nwint)
            ixint(nwint, ii) = xw(nwd)
            iyint(nwint, ii) = yw(nwd)
            izint(nwint, ii) = zw(nwd)
        end if
        !
        ! Return for new well data:
        !
        go to 5
        6          close(lin)
        write(ldbg, 109) nwd, nwint
        write(*, 109) nwd, nwint
        109        format(/, ' Number of acceptable well data  = ', i8, /, &
                ' Number of intervals for pert.   = ', i8)
        wfac = 20 / max(real(nwd), 1.0)
        !
        ! Now, establish the well intervals for the perturbation mechanism:
        !
        iwold = -1
    endif
    !
    ! Read the vertical proportion curve (if the file exists):
    !
    do iz = 1, nz
        do i = 1, 3
            pcurve(iz, i) = 0.0
        end do
    end do
    inquire(file = pcurvefl, exist = testfl)
    if(testfl.and.lvert) then
        write(*, *)
        write(*, *) 'Reading vertical proportion curve data'
        open(lin, file = pcurvefl, status = 'OLD')
        read(lin, *, err = 97)
        read(lin, *, err = 97) nvari
        do i = 1, nvari
            read(lin, *, err = 97)
        end do
        do i = 1, 3
            proptest(i) = 0.0
        end do
        do iz = 1, nz
            do i = 1, 3
                pcurve(iz, i) = 0.0
            end do
            read(lin, *, err = 97) (var(i), i = 1, nvari)
            if(lvertng) then
                ii = icolv(1)
                pcurve(iz, 1) = var(ii)
                proptest(1) = proptest(1) + pcurve(iz, 1)
            else
                do i = 1, 3
                    ii = icolv(i)
                    if(i.eq.1) pcurve(iz, i) = var(ii)
                    if(i.eq.2.and.ilev.eq.1)&
                            pcurve(iz, i) = var(ii)
                    if(i.eq.3.and.icre.eq.1)&
                            pcurve(iz, i) = var(ii)
                    proptest(i) = proptest(i) + pcurve(iz, i)
                end do
            end if
        end do
        close(lin)
        do i = 1, 3
            proptest(i) = proptest(i) / real(nz)
            if(proptest(i).lt.0.001) proptest(i) = 0.001
        end do
        do iz = 1, nz
            if(lvertng) then
                pcurve(iz, 1) = pcurve(iz, 1) * &
                        (tarprop(1) + tarprop(3) + tarprop(4)) / proptest(1)
            else
                pcurve(iz, 1) = pcurve(iz, 1) * &
                        tarprop(1) / proptest(1)
                if(ilev.eq.1)  pcurve(iz, 2) = pcurve(iz, 2) * &
                        tarprop(3) / proptest(2)
                if(icre.eq.1)  pcurve(iz, 3) = pcurve(iz, 3) * &
                        tarprop(4) / proptest(2)
            end if
        end do
    end if
    !
    ! Read the areal proportion map (if the file exists):
    !
    do iy = 1, ny
        do ix = 1, nx
            do i = 1, 3
                pmap(ix, iy, i) = 0.0
            end do
        end do
    end do
    inquire(file = pmapfl, exist = testfl)
    if(testfl.and.larea) then
        write(*, *)
        write(*, *) 'Reading areal proportion map data'
        open(lin, file = pmapfl, status = 'OLD')
        read(lin, *, err = 97)
        read(lin, *, err = 97) nvari
        do i = 1, nvari
            read(lin, *, err = 97)
        end do
        do i = 1, 3
            proptest(i) = 0.0
        end do
        do iy = 1, ny
            do ix = 1, nx
                do i = 1, 3
                    pmap(ix, iy, i) = 0.0
                end do
                read(lin, *, err = 97) (var(i), i = 1, nvari)
                if(lareang) then
                    ii = icola(1)
                    pmap(ix, iy, 1) = var(ii)
                    proptest(1) = proptest(1) + pmap(ix, iy, 1)
                else
                    do i = 1, 3
                        ii = icola(i)
                        if(i.eq.1) pmap(ix, iy, i) = var(ii)
                        if(i.eq.2.and.ilev.eq.1)&
                                pmap(ix, iy, i) = var(ii)
                        if(i.eq.3.and.icre.eq.1)&
                                pmap(ix, iy, i) = var(ii)
                        proptest(i) = proptest(i) + pmap(ix, iy, i)
                    end do
                end if
            end do
        end do
        close(lin)
        do i = 1, 3
            proptest(i) = proptest(i) / real(nx * ny)
            if(proptest(i).lt.0.001) proptest(i) = 0.001
        end do
        do iy = 1, ny
            do ix = 1, nx
                if(lareang) then
                    pmap(ix, iy, 1) = pmap(ix, iy, 1) * &
                            (tarprop(1) + tarprop(3) + tarprop(4)) / proptest(1)
                else
                    pmap(ix, iy, 1) = pmap(ix, iy, 1) * &
                            tarprop(1) / proptest(1)
                    if(ilev.eq.1) pmap(ix, iy, 2) = pmap(ix, iy, 2) * &
                            tarprop(3) / proptest(2)
                    if(icre.eq.1) pmap(ix, iy, 3) = pmap(ix, iy, 3) * &
                            tarprop(4) / proptest(2)
                end if
            end do
        end do
    end if
    !
    ! Open the output files and write header information:
    !
    open(lout, file = outfl, status = 'UNKNOWN')
    !      write(lout,110)
    ! 110  format('FLUVSIM Realizations',/,'1',/,'facies code')

    open(lgeo, file = geofl, status = 'UNKNOWN')
    write(lgeo, 130) nx, xmn, xsiz, ny, ymn, ysiz, nz, avgthick, ilev, icre
    130  format(i3, 1x, f10.2, 1x, f6.3, ' -X: nx, xmn, xsiz', /, &
            i3, 1x, f10.2, 1x, f6.3, ' -Y: ny, ymn, ysiz', /, &
            i3, 12x, f6.3, ' -Z: number and average thickness', /, &
            i2, i2, 17x, ' -levee and crevasse (0=no, 1=yes)')

    if(lvert) then
        open(lpv, file = pcurout, status = 'UNKNOWN')
        write(lpv, 111)
        111        format('FLUVSIM Vertical Proportion Curve Output', /, '7', /, &
                'Z index', /, 'target 1', /, 'actual 1', /, &
                'target 2', /, 'actual 2', /, &
                'target 3', /, 'actual 3')
    end if

    if(larea) then
        open(lpa, file = pmapout, status = 'UNKNOWN')
        write(lpa, 112)
        112        format('FLUVSIM Areal Proportion Map Output', /, '6', /, &
                'target 1', /, 'actual 1', /, &
                'target 2', /, 'actual 2', /, &
                'target 3', /, 'actual 3')
    end if

    if(lwell) then
        open(lwd, file = wellout, status = 'UNKNOWN')
        write(lwd, 113)
        113        format('FLUVSIM Well Data Output', /, '5', /, &
                'x', /, 'y', /, 'z', /, 'well data', /, 'realization')
    end if

    return
    !
    ! Error in an Input File Somewhere:
    !
    97   stop 'ERROR in proportion files!'
    98   stop 'ERROR in parameter file!'
    99   stop 'ERROR in data file!'
end


subroutine fluvsim
    !-----------------------------------------------------------------------
    !
    ! Main subroutine that establishes the initial set of channels (and
    ! associated levee and crevasse sands), perturbs them until one of the
    ! stopping criteria is met, and then writes output files.
    !
    !
    !
    !
    !-----------------------------------------------------------------------
    use       geostat
    real*8    acorni
    logical   accept, lreport, flong
    !
    ! ACORN parameters:
    !
    parameter(KORDEI = 12, MAXOP1 = KORDEI + 1, MAXINT = 2**30)
    common /iaco/ ixv(MAXOP1)
    !
    ! Initialize:
    !
    flong = .false.
    do ic = 1, mxc
        chanon(ic) = .false.
    end do
    nc = 0
    modprop(0) = 1.0
    do i = 1, 4
        modprop(i) = 0.0
    end do
    !
    ! Get 10 channels to calibrate the size of levees and frequency of
    ! crevasses.
    !
    ! USE SPECIFIED DIMENSIONS REGARDLESS OF PROPORTIONS
    !      if(ilev.eq.1.or.icre.eq.1) then
    !            write(*,*)
    !            write(*,*) 'Getting a set of channels to calibrate size of',
    !     +                 ' levees and crevasses'
    !            do i=1,10
    !                  call getchpar(i,flong)
    !            end do
    !            call rasterc
    !            if(ilev.eq.1) then
    !                  if(tarprop(1).gt.0.001.and.modprop(3).gt.0.001) then
    !                  sclf = tarprop(3)/tarprop(1) / (modprop(3)/modprop(1))
    !                  write(*,*)
    !                  write(*,*) '   Levee width: ',flw(2)
    !                  write(*,*) '   Scaled   by: ',sclf
    !                  if(lvert.and..not.lvertng) then
    !                        do iz=1,nz
    !                              sclf = pcurve(iz,2)/pcurve(iz,1)
    !     +                             /  (modprop(3)/modprop(1))
    !                              flwz(iz,1) = flw(1) * sclf
    !                              flwz(iz,2) = flw(2) * sclf
    !                              flwz(iz,3) = flw(3) * sclf
    !                        end do
    !                  else
    !                        flw(1) = flw(1) * sclf
    !                        flw(2) = flw(2) * sclf
    !                        flw(3) = flw(3) * sclf
    !                        write(*,*) '   New   width: ',flw(2)
    !                  end if
    !                  end if
    !            end if
    !            if(icre.eq.1) then
    !                  if(tarprop(1).gt.0.001.and.modprop(4).gt.0.001) then
    !                  sclf = tarprop(4)/tarprop(1) / (modprop(4)/modprop(1))
    !                  write(*,*)
    !                  write(*,*) '   Crevasse   Prob: ',creprob(2)
    !                  write(*,*) '   Scaled       by: ',sclf
    !                  if(lvert.and..not.lvertng) then
    !                        do iz=1,nz
    !                              sclf = pcurve(iz,3)/pcurve(iz,1)
    !     +                             /  (modprop(4)/modprop(1))
    !                              creprz(iz,1) = creprob(1) * sclf
    !                              creprz(iz,2) = creprob(2) * sclf
    !                              creprz(iz,3) = creprob(3) * sclf
    !                        end do
    !                  else
    !                        creprob(1) = creprob(1) * sclf
    !                        creprob(2) = creprob(2) * sclf
    !                        creprob(3) = creprob(3) * sclf
    !                        write(*,*) '   New probability:',creprob(2)
    !                  end if
    !                  end if
    !            end if
    !            write(*,*)
    !      end if
    !
    ! Reset the channel indicators for the starting set of channels:
    !
    do ic = 1, mxc
        chanon(ic) = .false.
    end do
    nc = 0
    modprop(0) = 1.0
    do i = 1, 4
        modprop(i) = 0.0
    end do
    !
    ! Get initial channels that honor well data (approximately):
    !
    !      write(*,*)
    !      write(*,*) 'Getting channels that honor well data'
    !      call getchw
    !      call rasterc
    nc = 0
    !      do ic=1,mxc
    !            if(chanon(ic)) nc = nc + 1
    !      end do
    write(*, *)
    write(*, *) 'Added ', nc, ' channels for well data'
    write(*, *) '  target proportion ', tarprop(0), ' at ', modprop(0)
    !
    ! Get a starting set of channels (until channel proportion is met):
    !
    write(*, *)
    write(*, *) 'Adding channels to get to net-to-gross'
    1    if(modprop(0).le.(1.05 * tarprop(0))) go to 2
    !     do =1,10  ORIOL
    do i = 1, 3
        nc = nc + 1
        if(nc.eq.mxc) go to 2
        write (*, *) 'Working on channel:', nc
        call getchpar(nc, flong)
    end do
    call rasterc
    write(*, 500) nc, modprop(1), modprop(3), modprop(4)
    500  format('   channel ', i3, ' channel: ', f6.4, &
            ' levee: ', f6.4, &
            ' crevasse: ', f6.4)
    go to 1
    2    continue
    !     do =1,10
    do i = 1, 3
        write (*, *) modprop
        write (*, *) tarprop
        !           check proportion of floodplain
        if(modprop(0).gt.tarprop(0)) go to 3
        chanon(nc) = .false.
        nc = nc - 1
        if(nc.le.1) go to 3
        call rasterc
    end do
    3    continue
    write(*, 500) nc, modprop(1), modprop(3), modprop(4)
    !
    ! Save a good copy of the channum and chanind arrays:
    !
    do iz = 1, nz
        do iy = 1, ny
            do ix = 1, nx
                chanind(ix, iy, iz) = tchanind(ix, iy, iz)
                channum(ix, iy, iz) = tchannum(ix, iy, iz)
            end do
        end do
    end do

    ! BYPASS SIMULATED ANNEALING, OF- MAR2017
    go to 5
    !
    ! Establish the initial objective function and scaling:
    !
    if(niter.le.0.or.objmin.ge.1.0) go to 5
    write(*, *)
    write(*, *) 'Starting to iterate set of channels'
    if(lwell) write(*, 502)
    502  format(/, '     Note that a well mismatch of 10% is considered ', &
            'good.', /)
    lreport = .true.
    call getobj(objcur, lreport)
    wellmis = welltry
    lreport = .false.
    objscl = objcur
    if(objcur.lt.EPSLON) go to 5
    objscl = 1.0 / objscl
    write(*, 99) 0, objcur * objscl, wellmis
    !
    ! MAIN Loop until convergence:
    !
    npert = 0
    iend = 0
    temp = t0
    nnochange = 0
    10   naccept = 0
    ntry = 0
    !
    ! Keep perturbing system until we exceed some limit:
    !
    20   ntry = ntry + 1
    npert = npert + 1
    nnochange = nnochange + 1
    !
    ! Perturb the set of fluvial geo-objects, get a raster image, and
    ! update the objective function:
    !
    call getpert
    call rasterc
    call getobj(objtry, lreport)
    !
    ! Simulated annealing-based rule to accept the perturbation:
    !
    accept = .false.
    if(objtry.ge.objcur) then
        if(temp.gt.0.0) then
            unif = max(EPSLON, real(acorni(idum)))
            if(objtry * objscl.lt.&
                    (objcur * objscl - dble(temp * alog(unif))))&
                    accept = .true.
        end if
    else
        accept = .true.
    endif
    !
    ! Accept perturbation: reset objective function and update a copy of
    !                      the chanind/channum arrays.
    !
    if(accept) then
        objcur = objtry
        wellmis = welltry
        nnochange = 0
        naccept = naccept + 1
        do iz = 1, nz
            do iy = 1, ny
                do ix = 1, nx
                    chanind(ix, iy, iz) = tchanind(ix, iy, iz)
                    channum(ix, iy, iz) = tchannum(ix, iy, iz)
                end do
            end do
        end do
    else
        !
        ! Reject perturbation: undo all channels turned on, turn on channels
        !                      that were turned off, and vertically shift those
        !                      channels that were moved
        !
        do i = 1, nonp
            ic = icon(i)
            chanon(ic) = .false.
        end do
        do i = 1, noffp
            ic = icoff(i)
            chanon(ic) = .true.
        end do
        do i = 1, nvshift
            ic = shiftc(i)
            if(chanon(ic)) then
                ndelz = -int(shiftp(i))
                call vshift(ic, ndelz)
            end if
        end do
        nc = 0
        do ic = 1, mxc
            if(chanon(ic)) nc = nc + 1
        end do
    end if
    !
    ! Report on status:
    !
    write(ldbg, '(i6,f10.6)') npert, objcur * objscl
    if(accept) then
        write(*, 100)       npert, objcur * objscl, wellmis
    else
        write(*, 101)       npert, objcur * objscl, wellmis
    end if
    99  format('     iteration ', i5, ' obj: ', f8.5, &
            '  well mismatch: ', f7.2, '%')
    100  format('     iteration ', i5, ' obj: ', f8.5, &
            '  well mismatch: ', f7.2, '%', &
            '  (accept)')
    101  format('     iteration ', i5, ' obj: ', f8.5, &
            '  well mismatch: ', f7.2, '%', &
            '  (reject)')
    !
    ! Are we finished yet?
    !
    if(objcur * objscl.lt.objmin.or.iend.ge.numred.or.&
            npert.eq.niter.or.nnochange.gt.mnoc) then
        if(objcur * objscl.lt.objmin)       write(*, 401)
        if(iend.ge.numred)                write(*, 402)
        if(npert.eq.niter)                write(*, 403)
        if(nnochange.gt.mnoc)             write(*, 404)
        401        format(' Stopped because of obj lt objmin')
        402        format(' Stopped because of iend gt num')
        403        format(' Stopped because of npert gt niter')
        404        format(' Stopped because of number of perturbations without'&
                , ' a change')
        go to 5
    endif
    !
    ! Tried too many at this "temperature"?
    !
    if(ntry.gt.kasas.and.temp.gt.1.0e-16) then
        iend = iend + 1
        temp = redfac * temp
        write(*, 430) temp
        go to 10
    endif
    !
    ! Accepted enough at this "temperature"?
    !
    if(naccept.ge.ksas.and.temp.gt.1.0e-16) then
        temp = redfac * temp
        write(*, 430) temp
        iend = 0
        go to 10
    endif
    430  format('  lowering temperature to ', f10.8)
    !
    ! Go back for another attempted swap:
    !
    go to 20
    !
    ! Finished with this realization:
    !
    5    continue
    lreport = .true.
    call getobj(objtry, lreport)
    write(*, 100) npert, objcur * objscl, wellmis
    write(*, 500) nc, modprop(1), modprop(3), modprop(4)
    write(ldbg, '(i6,f10.6)') npert, objcur * objscl
    !
    ! Write this realization to the output file:
    !
    write(*, *)
    write(*, *) 'Writing this realization to output files'
    do i = 0, 4
        modprop(i) = 0.0
    end do
    do iz = 1, nz
        do iy = 1, ny
            do ix = 1, nx
                icode = chanind(ix, iy, iz)
                !
                ! Determine if this is an "edge" cell:
                ! REMOVE CHANNEL BASES
                !            if(icode.eq.1) then
                !                  ichan = channum(ix,iy,iz)
                !                  ixl   = max((ix-1), 1)
                !                  ixu   = min((ix+1),nx)
                !                  iyl   = max((iy-1), 1)
                !                  iyu   = min((iy+1),ny)
                !                  izl   = max((iz-1), 1)
                !                  if(channum(ixl,iy ,iz ).lt.ichan) icode = 2
                !                  if(channum(ixu,iy ,iz ).lt.ichan) icode = 2
                !                  if(channum(ix ,iyl,iz ).lt.ichan) icode = 2
                !                  if(channum(ix ,iyu,iz ).lt.ichan) icode = 2
                !                  if(channum(ix ,iy ,izl).lt.ichan) icode = 2
                !            end if
                modprop(icode) = modprop(icode) + 1.
                !
                ! Write this cell to file
                !
                write(lout, '(i1)')       icode
                !           write(lout,'(i1,1x,i4)') icode,channum(ix,iy,iz)
            end do
        end do
    end do
    do i = 0, 4
        modprop(i) = modprop(i) / real(nx * ny * nz)
    end do
    write(*, 600) modprop(0), tarprop(0), &
            modprop(1) + modprop(2), tarprop(1), &
            modprop(3), tarprop(3), &
            modprop(4), tarprop(4)
    600  format(/, ' Proportion of floodplain shale = ', f6.4, &
            ' (target = ', f6.4, ')', /, &
            ' Proportion of channel sand     = ', f6.4, &
            ' (target = ', f6.4, ')', /, &
            ' Proportion of levee sand       = ', f6.4, &
            ' (target = ', f6.4, ')', /, &
            ' Proportion of crevasse sand    = ', f6.4, &
            ' (target = ', f6.4, ')', /)
    !
    ! Write the geometry data:
    !
    do ic = 1, mxc
        if(chanon(ic)) then
            write(lgeo, 511) ic, cz(ic), nyc(ic)
            511              format(' Channel ', i3, ' Z ', f6.1, ' ny ', i4)
            do iy = 1, nyc(ic)
                write(lgeo, 512)cx(ic, iy, 4), cw(ic, iy), ct(ic, iy, 4)
                512                    format(12(f12.3, 1x))
            end do
        end if
    end do
    !
    ! Write the proportion data:
    !
    if(lvert) then
        do iz = 1, nz
            write(lpv, 515) iz, pcurve(iz, 1), pcurvea(iz, 1), &
                    pcurve(iz, 2), pcurvea(iz, 2), &
                    pcurve(iz, 3), pcurvea(iz, 3)
            515              format(i3, 1x, 3(1x, f7.4, 1x, f7.4))
        end do
    end if
    if(larea) then
        do iy = 1, ny
            do ix = 1, nx
                write(lpa, 516) pmap(ix, iy, 1), pmapa(ix, iy, 1), &
                        pmap(ix, iy, 2), pmapa(ix, iy, 2), &
                        pmap(ix, iy, 3), pmapa(ix, iy, 3)
                516                    format(3(1x, f7.4, 1x, f7.4))
            end do
        end do
    end if
    !
    ! Write the well data:
    !
    if(lwell) then
        do i = 1, nwd
            ix = xw(i)
            iy = yw(i)
            iz = zw(i)
            ii = 0
            if(channum(ix, iy, iz).gt.0) ii = 1
            write(lwd, 520) xw(i), yw(i), zw(i), fw(i), ii, &
                    channum(ix, iy, iz)
            520              format(3i4, 1x, 2i2, 1x, i4)
        end do
        write(*, *)
        write(*, *) ' Percent Mismatch at Well = ', wellmis
    end if
    !
    ! Return to the main program:
    !
    return
end


subroutine getchpar(ic, flong)
    !-----------------------------------------------------------------------
    !
    ! Establish a set of legitimate parameters for channel "ic" given input
    ! triangular distributions
    !
    !
    !
    !-----------------------------------------------------------------------
    use       geostat
    real*8    acorni
    real      testx(4), testy(4)
    logical   testpt(4), flong, inflag
    !
    ! ACORN parameters:
    !
    parameter(KORDEI = 12, MAXOP1 = KORDEI + 1, MAXINT = 2**30)
    common /iaco/ ixv(MAXOP1)
    !
    ! Turn channel on and get Z position:
    !
    if(ic.le.0.or.ic.gt.mxc) ic = 1
    chanon(ic) = .true.
    cz(ic) = real(acorni(idum))
    !
    ! Get orientation angle within -45 to 135 and not exactly 90.0:
    !
    cang(ic) = getval(fco)
    1    if(cang(ic).gt.135.0) cang(ic) = cang(ic) - 180.0
    if(cang(ic).lt.-45.0) cang(ic) = cang(ic) + 180.0
    if(cang(ic).lt.-45.0.or.cang(ic).gt.135.0) go to 1
    if(abs(cang(ic) - 90.0).lt.0.001) cang(ic) = 89.99
    ccosa(ic) = cos(cang(ic) * DEGTOR)
    csina(ic) = sin(cang(ic) * DEGTOR)
    ctana(ic) = tan(cang(ic) * DEGTOR)
    if(abs(ctana(ic)).lt.0.001) ctana(ic) = 0.001
    !
    ! Establish a first estimate of the origin of the channel line depending
    ! on whether the channel is aligned closer to the X or Y axis:
    !
    if(abs(csina(ic)).gt.abs(ccosa(ic))) then
        cxorigin(ic) = cxmin
        if(ccosa(ic).le.0.0) then
            cyl = cymin
            cyu = cxmax - ccosa(ic) * (cxmax - cxmin)
        else
            cyl = cymin - 0.5 * ysiz - ccosa(ic) * (xmx - cxmin)
            cyu = cymax
        end if
        cyorigin(ic) = cyl + real(acorni(idum)) * (cyu - cyl)
    else
        cyorigin(ic) = cymin
        if(csina(ic).le.0.0) then
            cxl = cxmin
            cxu = cxmax - csina(ic) * (ymx - cymin)
        else
            cxl = cxmin - 0.5 * xsiz - csina(ic) * (ymx - cymin)
            cxu = cxmax
        end if
        cxorigin(ic) = cxl + real(acorni(idum)) * (cxu - cxl)
    end if
    !
    ! Settle on an origin (as close as possible to the area of interest) and
    ! length (nyc) that keeps channel in area of interest:
    !
    testx(1) = cxmin
    testy(1) = cyorigin(ic) - (cxorigin(ic) - testx(1)) / ctana(ic)
    testpt(1) = .false.
    if(testy(1).ge.cymin.and.testy(1).le.cymax) testpt(1) = .true.

    testx(2) = cxmax
    testy(2) = cyorigin(ic) - (cxorigin(ic) - testx(2)) / ctana(ic)
    testpt(2) = .false.
    if(testy(2).ge.cymin.and.testy(2).le.cymax) testpt(2) = .true.

    testy(3) = cymin
    testx(3) = cxorigin(ic) - (cyorigin(ic) - testy(3)) * ctana(ic)
    testpt(3) = .false.
    if(testx(3).ge.cxmin.and.testx(3).le.cxmax) testpt(3) = .true.

    testy(4) = cymax
    testx(4) = cxorigin(ic) - (cyorigin(ic) - testy(4)) * ctana(ic)
    testpt(4) = .false.
    if(testx(4).ge.cxmin.and.testx(4).le.cxmax) testpt(4) = .true.
    !
    ! Get origin that is in the grid (recall that the origin may be outside
    ! to allow a sinuous channel to cut the edge of the model):
    !
    testmax = 1.0e20
    iorigin = 1
    do i = 1, 4
        if(testpt(i)) then
            testdis = (cxorigin(ic) - testx(i))**2&
                    + (cyorigin(ic) - testy(i))**2
            if(testdis.lt.testmax) then
                iorigin = i
                testmax = testdis
            end if
        end if
    end do
    cxorigin(ic) = testx(iorigin)
    cyorigin(ic) = testy(iorigin)
    !
    ! Establish how long the channel must be to keep it in the grid:
    !
    testmax = 0.0
    do i = 1, 4
        if(testpt(i)) then
            testlen = sqrt((cxorigin(ic) - testx(i))**2 + &
                    (cyorigin(ic) - testy(i))**2)
            if(testlen.gt.testmax) testmax = testlen
        end if
    end do
    nyc(ic) = testmax / cdy
    if(nyc(ic).lt.10)   nyc(ic) = 10
    if(nyc(ic).gt.MAXL) then
        write(*, *) 'WARNING: attempting a channel ', nyc(ic), 'long'
        write(*, *) '                    only have ', MAXL
        nyc(ic) = MAXL
    end if
    !
    ! Force the channel to be long? (because it may get moved around)
    !
    if(flong) nyc(ic) = MAXL
    !
    ! Channel center line:
    !
    range = getval(fcal)
    avgdep = getval(fcad)
    call get1d(nyc(ic), cdy, range, rarray)
    do iy = 1, nyc(ic)
        ccent(ic, iy) = rarray(iy) * avgdep
        chx(iy) = ccent(ic, iy)
        !	    write (*,*) 'chx', chx(iy)
    end do
    !
    ! Channel thickness:
    !
    avgthick = getval(fct)
    avgdep = getval(fctu)
    if(avgdep.ge.0.95.and.avgdep.le.1.05) then
        do iy = 1, nyc(ic)
            rarray(iy) = 0.0
        end do
    else
        range = getval(fctul)
        call get1d(nyc(ic), cdy, range, rarray)
    end if
    do iy = 1, nyc(ic)
        ct(ic, iy, 1) = 0.0
        ct(ic, iy, 4) = avgthick * (1.0 + rarray(iy)) * avgdep
        ct(ic, iy, 7) = 0.0
    end do
    !
    ! Channel width:
    !
    cwtr = getval(fcwt)
    avgdep = getval(fcwu)
    if(avgdep.ge.0.95.and.avgdep.le.1.05) then
        do iy = 1, nyc(ic)
            rarray(iy) = 0.0
        end do
    else
        range = getval(fcwul)
        call get1d(nyc(ic), cdy, range, rarray)
    end if
    do iy = 1, nyc(ic)
        width = (1.0 + rarray(iy)) * cwtr * ct(ic, iy, 4)
        cx(ic, iy, 1) = chx(iy) - 0.5 * width
        cx(ic, iy, 7) = chx(iy) + 0.5 * width
    end do
    !
    ! Channel curvature and relative position (of thickness maximum):
    !
    do i = 1, nyc(ic) - 2
        tt = sqrt(cdy * cdy + (chx(i + 1) - chx(i))**2)
        p1 = atan((chx(i + 1) - chx(i)) / cdy)
        p2 = atan((chx(i + 2) - chx(i + 1)) / cdy)
        rarray(i) = (p2 - p1) / max(tt, EPSLON)
        ! RESET CURVATURE - ORIOL 2017
        rarray(i) = 0
        !           write (*,*) rarray(i)
    end do
    rarray(nyc(ic) - 1) = rarray(nyc(ic) - 2)
    rarray(nyc(ic)) = rarray(nyc(ic) - 2)
    curmaxn = -1.0e20
    curmaxp = -1.0e20
    do i = 1, nyc(ic)
        sval = 0.0
        snor = 0.0
        do j = -10, 10
            k = i + j
            if(k.ge.1.and.k.le.nyc(ic)) then
                swgt = 1.0 / real(4 + abs(j))
                sval = sval + rarray(k) * swgt
                snor = snor + swgt
            end if
        end do
        chx(i) = sval / snor
        if(chx(i).lt.0.0) then
            if(abs(chx(i)).gt.curmaxn) curmaxn = abs(chx(i))
        else
            if(chx(i).gt.curmaxp) curmaxp = chx(i)
        end if
    end do
    curmaxp = max((2.5 * curmaxp), EPSLON)
    curmaxn = max((2.5 * curmaxn), EPSLON)
    do i = 1, nyc(ic)
        if(abs(0.5 - chx(i)).eq.EPSLON) then
            chx(i) = 0.5
        else if(chx(i).lt.0.0) then
            chx(i) = 0.5 * (1.0 + abs(chx(i)) / curmaxn)
        else
            chx(i) = 0.5 * (1.0 - chx(i) / curmaxp)
        end if
        !	    write (*,*) 'chx', chx(i)
    end do
    !
    ! Now, finish the channel cross section arrays cx and ct using chx as
    ! the position of maximum thickness:
    !
    do iy = 1, nyc(ic)

        cx(ic, iy, 4) = cx(ic, iy, 1) + chx(iy) * (cx(ic, iy, 7) - cx(ic, iy, 1))

        cx(ic, iy, 2) = cx(ic, iy, 1) + 1.0 / 3.0 * (cx(ic, iy, 4) - cx(ic, iy, 1))
        call csect(cx(ic, iy, 1), cx(ic, iy, 7), chx(iy), ct(ic, iy, 4), &
                cx(ic, iy, 2), ct(ic, iy, 2))
        cx(ic, iy, 3) = cx(ic, iy, 1) + 2.0 / 3.0 * (cx(ic, iy, 4) - cx(ic, iy, 1))
        call csect(cx(ic, iy, 1), cx(ic, iy, 7), chx(iy), ct(ic, iy, 4), &
                cx(ic, iy, 3), ct(ic, iy, 3))

        cx(ic, iy, 5) = cx(ic, iy, 4) + 1.0 / 3.0 * (cx(ic, iy, 7) - cx(ic, iy, 4))
        call csect(cx(ic, iy, 1), cx(ic, iy, 7), chx(iy), ct(ic, iy, 4), &
                cx(ic, iy, 5), ct(ic, iy, 5))
        cx(ic, iy, 6) = cx(ic, iy, 4) + 2.0 / 3.0 * (cx(ic, iy, 7) - cx(ic, iy, 4))
        call csect(cx(ic, iy, 1), cx(ic, iy, 7), chx(iy), ct(ic, iy, 4), &
                cx(ic, iy, 6), ct(ic, iy, 6))

    end do
    !
    ! Build the template for this channel:
    !
    call getindx(nz, zmn, zsiz, cz(ic), izhi, inflag)
    izhitem(ic) = izhi
    do iy = 1, nyc(ic)
        if(cx(ic, iy, 1).lt.0.0) then
            ixlotem(ic, iy) = int(cx(ic, iy, 1) / cdy - 0.5)
        else
            ixlotem(ic, iy) = int(cx(ic, iy, 1) / cdy + 0.5)
        end if
        if(cx(ic, iy, 7).lt.0.0) then
            ixhitem(ic, iy) = int(cx(ic, iy, 7) / cdy - 0.5)
        else
            ixhitem(ic, iy) = int(cx(ic, iy, 7) / cdy + 0.5)
        end if

        if(ixlotem(ic, iy).lt.-MAXW.or.ixhitem(ic, iy).gt.MAXW) then
            write(*, *) 'WARNING: channel too wide '
            if(ixlotem(ic, iy).lt.-MAXW) ixlotem(ic, iy) = -MAXW
            if(ixhitem(ic, iy).gt. MAXW) ixhitem(ic, iy) = MAXW
        end if

        zbot = cz(ic) - ct(ic, iy, 4)
        call getindx(nz, zmn, zsiz, zbot, izbot, inflag)

        xx = real(ixlotem(ic, iy) - 1) * cdy
        do ix = ixlotem(ic, iy), ixhitem(ic, iy)

            xx = xx + cdy
            izlotem(ic, iy, ix) = izhi + 1

            if(xx.gt.cx(ic, iy, 1).and.xx.lt.cx(ic, iy, 7)) then

                do i = 2, 7
                    if(xx.le.cx(ic, iy, i)) then
                        zl = cz(ic) - (ct(ic, iy, i - 1) + &
                                (xx - cx(ic, iy, i - 1)) / (cx(ic, iy, i) - cx(ic, iy, i - 1))&
                                        * (ct(ic, iy, i) - ct(ic, iy, i - 1)))
                        go to 2
                    end if
                end do
                2                continue

                if(zl.gt.1.0) then
                    izlotem(ic, iy, ix) = izhi + 1
                else
                    call getindx(nz, zmn, zsiz, zl, izlo, inflag)
                    izlotem(ic, iy, ix) = izlo
                end if

            end if

        end do

    end do
    !
    ! Do we need to build levee template?
    !

    if(ilev.eq.1) then
        range = 0.5 * real(nyc(ic)) * ysiz
        !	    write (*,*) range, nyc(ic)
        if(lvert.and..not.lvertng) then
            iiz = izhitem(ic)
            flw(1) = flwz(iiz, 1)
            flw(2) = flwz(iiz, 2)
            flw(3) = flwz(iiz, 3)
        end if
        awidth = getval(flw)
        htfac = getval(flh)
        redfac = getval(fld)
        call get1d(nyc(ic), cdy, range, rarray)
        do iy = 1, nyc(ic)
            dep = cz(ic) - redfac * ct(ic, iy, 4)
            call getindx(nz, zmn, zsiz, dep, idep, inflag)
            llzlo(ic, iy) = idep
            llwid(ic, iy) = 1 + int((awidth + 0.25 * rarray(iy)) / cdy)
            if(llwid(ic, iy).gt.MAXBW) llwid(ic, iy) = MAXBW
            wid = real(llwid(ic, iy))
            nbw = min(llwid(ic, iy), MAXBW)
            ht = htfac * ct(ic, iy, 4)
            !                  write (*,*) awidth
            !                  write (*,*) wid
            !                  write (*,*) nbw
            do ix = 1, nbw
                call csect(0.0, wid, 0.75, ht, real(ix), thick)
                dep = cz(ic) + thick
                !			write (*,*),dep, thick
                call getindx(nz, zmn, zsiz, dep, idep, inflag)
                llbh(ic, iy, ix) = idep
                !			write (*,*),idep
            end do
        end do

        awidth = getval(flw)
        htfac = getval(flh)
        redfac = getval(fld)
        call get1d(nyc(ic), cdy, range, rarray)
        do iy = 1, nyc(ic)
            dep = cz(ic) - redfac * ct(ic, iy, 4)
            call getindx(nz, zmn, zsiz, dep, idep, inflag)
            rlzlo(ic, iy) = idep
            rlwid(ic, iy) = 1 + int((awidth + 0.25 * rarray(iy)) / cdy)
            if(rlwid(ic, iy).gt.MAXBW) rlwid(ic, iy) = MAXBW
            wid = real(rlwid(ic, iy))
            nbw = min(rlwid(ic, iy), MAXBW)
            ht = htfac * ct(ic, iy, 4)
            do ix = 1, nbw
                call csect(0.0, wid, 0.25, ht, real(ix), thick)
                dep = cz(ic) + thick
                call getindx(nz, zmn, zsiz, dep, idep, inflag)
                ! changed for asmetry in levee ' ORIOL 2017
                rlbh(ic, iy, nbw - ix + 1) = idep
            end do
        end do

    end if
    !
    ! Consider adding crevasse splays?
    !

    if(icre.eq.1) then
        if(lvert.and..not.lvertng) then
            iiz = izhitem(ic)
            creprob(1) = creprz(iiz, 1)
            creprob(2) = creprz(iiz, 2)
            creprob(3) = creprz(iiz, 3)
        end if
        ncre(ic) = int(getval(creprob) + 0.5)
        if(creprob(3).lt.1.0.and.&
                real(acorni(idum)).lt.creprob(2)) ncre(ic) = 1
        if(ncre(ic).gt.MAXCRE) ncre(ic) = MAXCRE
        if(ncre(ic).gt.0) then
            !
            ! Get a 1-D array of crevasse occurence probability along the channel:
            !
            sumprob = 0.0
            do iy = 1, nyc(ic)
                rarray(iy) = 0.25 - chx(iy) * (1.0 - chx(iy))
                sumprob = sumprob + rarray(iy)
            end do
            sumprob = 1.0 / sumprob
            rarray(1) = rarray(1) * sumprob
            do iy = 2, nyc(ic)
                rarray(iy) = rarray(iy - 1) + rarray(iy) * sumprob
            end do
        end if
        !
        ! Establish the locations of the crevasses:
        !
        do icrev = 1, ncre(ic)
            cdf = real(acorni(idum))
            plo = 0.0
            do iy = 1, nyc(ic)
                if(cdf.ge.plo.and.cdf.le.rarray(iy)) then
                    jy = iy
                    go to 4
                end if
            end do
            4                cryloc(ic, icrev) = jy
            if(chx(jy).lt.0.5) then
                crright(ic, icrev) = .true.
                crxloc(ic, icrev) = ixhitem(ic, jy) - 1
            else
                crright(ic, icrev) = .false.
                crxloc(ic, icrev) = ixlotem(ic, jy)
            end if
            crnum(ic, icrev) = 1 + int(real(acorni(idum)) * MAXCRE)
            crt(ic, icrev) = getval(fcrt)
        end do
    end if
    !
    ! Finished assigning parameters for this channel:
    !
    return
end


subroutine rasterc
    !-----------------------------------------------------------------------
    !
    ! Go through all channels and their associated levees and crevasses
    ! creating a raster image of the lithofacies types
    !
    !
    !
    !-----------------------------------------------------------------------
    use       geostat
    parameter(DEG2RAD = 3.141592654 / 180.0)
    real      tmpcn(1000), tmpcz(1000)
    logical   inflag
    !
    ! Get time order of channels:
    !
    nc = 0
    do ic = 1, mxc
        !	    write(*,*) chanon (ic)
        if(chanon(ic)) then
            nc = nc + 1
            tmpcn(nc) = real(ic)
            tmpcz(nc) = cz(ic)
        end if
    end do
    call sortem(1, nc, tmpcz, 1, tmpcn, c, d, e, f, g, h)
    !
    ! Reset channel arrays:
    !
    do iz = 1, nz
        do iy = 1, ny
            do ix = 1, nx
                tchanind(ix, iy, iz) = 0
                tchannum(ix, iy, iz) = 0
            end do
        end do
    end do
    !
    ! Loop over all channels:
    !
    do iloopc = 1, nc
        ic = int(tmpcn(iloopc) + 0.5)
        !	    write (*,*) ic
        do iy = 1, nyc(ic)
            do ix = ixlotem(ic, iy), ixhitem(ic, iy)
                !
                ! Get position in "real" coordinates (not channel coordinates):
                !
                yalongc = real(iy - 1) * cdy
                xalongc = real(ix) * cdy
                yy = -csina(ic) * xalongc + ccosa(ic) * yalongc&
                        + cyorigin(ic)
                call getindx(ny, ymn, ysiz, yy, iiy, inflag)
                if(.not.inflag) go to 30
                xx = ccosa(ic) * xalongc + csina(ic) * yalongc&
                        + cxorigin(ic)
                call getindx(nx, xmn, xsiz, xx, iix, inflag)
                if(.not.inflag) go to 30
                !
                ! Code this stack if it is in the grid:
                !
                do iz = izlotem(ic, iy, ix), izhitem(ic)
                    tchanind(iix, iiy, iz) = 1
                    tchannum(iix, iiy, iz) = ic
                end do
                !		        if (iy.eq.50) write (*,*),ix,iiy,iix,izlotem(ic,iy,ix),izhitem(ic)

                30                     continue
            end do
            !
            ! Add levees?
            !
            if(ilev.eq.1) then
                !
                ! Left side:
                !
                ix = min((ixlotem(ic, iy) - llwid(ic, iy) - 1), MAXBW)
                ixfinish = (ixlotem(ic, iy) + ixhitem(ic, iy)) / 2
                nloop = ixfinish - ix
                !		  write (*,*) 'ix,ixfinish,nloop', ix,ixfinish,nloop
                do iloop = 1, nloop
                    ix = ix + 1
                    !
                    ! - get position in "real" coordinates (not channel coordinates):
                    !
                    yalongc = real(iy - 1) * cdy
                    xalongc = real(ix) * cdy
                    yy = -csina(ic) * xalongc + ccosa(ic) * yalongc&
                            + cyorigin(ic)
                    call getindx(ny, ymn, ysiz, yy, iiy, inflag)
                    if(.not.inflag) go to 31
                    xx = ccosa(ic) * xalongc + csina(ic) * yalongc&
                            + cxorigin(ic)
                    call getindx(nx, xmn, xsiz, xx, iix, inflag)
                    if(.not.inflag) go to 31
                    !
                    ! - get Z limits:
                    !
                    !                        iztop = izhitem(ic)
                    !                        kx    = min(iloop,MAXBW)
                    !                        if(iloop.le.llwid(ic,iy))
                    !     +                  iztop = llbh(ic,iy,kx)
                    !                        izbot = iztop + 1
                    !                        kx = min((ix + llwid(ic,iy)),ixhitem(ic,iy))
                    !                        do iz=llzlo(ic,iy),izhitem(ic)
                    !                              if(izlotem(ic,iy,kx).le.iz) then
                    !                                    izbot = iz
                    !                                    go to 10
                    !                              end if
                    !                        end do

                    ! new levees
                    !                        if (iloop.le.10)

                    10                     continue
                    !			izbot = izhitem(ic) -10
                    ! flat bottom - ORIOL 2017
                    izbot = llzlo(ic, iy)
                    !                       izbot = izlotem (ic,iy,iloop)
                    iztop = llbh(ic, iy, iloop)
                    !
                    ! - assign levee codes (if needed):
                    !
                    do iz = izbot, iztop
                        !                        do iz = iztop-5,iztop
                        if(iz.gt.0.0.and.iz.le.nz) then
                            if(tchannum(iix, iiy, iz).ne.ic) then
                                tchanind(iix, iiy, iz) = 3
                                tchannum(iix, iiy, iz) = -ic
                            end if
                        end if

                    end do
                    if ((iy.gt.50).AND.(iy.lt.55)) then
                        !			    write (*,*),iloop,ix,iiy,iix,izbot,iztop
                        !			    write (*,*),llwid (ic,iy), llbh(ic,iy,iloop), ixhitem(ic,iy)
                        !                            write (*,*),llzlo(ic,iy),izhitem(ic)
                    end if

                    31               continue
                end do
                !
                ! Right side:
                !
                ix = ixhitem(ic, iy) + rlwid(ic, iy) + 1
                ixfinish = (ixlotem(ic, iy) + ixhitem(ic, iy)) / 2
                nloop = ix - ixfinish
                do iloop = 1, nloop
                    ix = ix - 1
                    !
                    ! - get position in "real" coordinates (not channel coordinates):
                    !
                    yalongc = real(iy - 1) * cdy
                    xalongc = real(ix) * cdy
                    yy = -csina(ic) * xalongc + ccosa(ic) * yalongc&
                            + cyorigin(ic)
                    call getindx(ny, ymn, ysiz, yy, iiy, inflag)
                    if(.not.inflag) go to 32
                    xx = ccosa(ic) * xalongc + csina(ic) * yalongc&
                            + cxorigin(ic)
                    call getindx(nx, xmn, xsiz, xx, iix, inflag)
                    if(.not.inflag) go to 32
                    !
                    ! - get Z limits:
                    !
                    !                        iztop = izhitem(ic)
                    !                        if(iloop.le.rlwid(ic,iy)) then
                    !                              kx    = min((rlwid(ic,iy)-iloop+1),MAXBW)
                    !                              iztop = rlbh(ic,iy,kx)
                    !                        end if
                    !                        izbot = iztop + 1
                    !                        kx = max((ix - rlwid(ic,iy)),ixlotem(ic,iy))
                    !                        do iz=rlzlo(ic,iy),izhitem(ic)
                    !                              if(izlotem(ic,iy,kx).le.iz) then
                    !                                    izbot = iz
                    !                                    go to 11
                    !                              end if
                    !                        end do
                    11                     continue

                    ! flat bottom - ORIOL 2017
                    izbot = rlzlo(ic, iy)
                    iztop = rlbh(ic, iy, iloop)
                    !
                    ! - assign levee codes (if needed):
                    !
                    do iz = izbot, iztop
                        if(iz.gt.0.0.and.iz.le.nz) then
                            if(tchannum(iix, iiy, iz).ne.ic) then
                                tchanind(iix, iiy, iz) = 3
                                tchannum(iix, iiy, iz) = -ic
                            end if
                        end if
                    end do
                    32               continue
                end do
                !
                ! Finished adding levees.
                !
            end if
            !
            ! Finished looping over y slices along channel.
            !
        end do
        !
        ! Consider adding crevasses:
        !
        if(icre.eq.1) then

            do icrev = 1, ncre(ic)
                kcre = crnum(ic, icrev)
                iz = izhitem(ic)
                iy = cryloc(ic, icrev)
                do jx = 0, MAXCRX
                    do jy = -MAXCRY, MAXCRY
                        if(cre(kcre, jx, jy)) then
                            iy = cryloc(ic, icrev) + jy
                            ix = crxloc(ic, icrev) - jx
                            if(crright(ic, icrev)) ix = crxloc(ic, icrev) + jx
                            !
                            ! - get position in "real" coordinates (not channel coordinates):
                            !
                            yalongc = real(iy - 1) * cdy
                            xalongc = real(ix) * cdy
                            yy = -csina(ic) * xalongc + ccosa(ic) * yalongc&
                                    + cyorigin(ic)
                            call getindx(ny, ymn, ysiz, yy, iiy, inflag)
                            if(.not.inflag) go to 1
                            xx = ccosa(ic) * xalongc + csina(ic) * yalongc&
                                    + cxorigin(ic)
                            call getindx(nx, xmn, xsiz, xx, iix, inflag)
                            if(.not.inflag) go to 1
                            !
                            ! - do we need to erase the levee (bank above channel) at this location?
                            !
                            if(ilev.eq.1) then
                                do iz = izhitem(ic), nz
                                    if(tchannum(iix, iiy, iz).eq.-ic) then
                                        tchannum(iix, iiy, iz) = 0
                                        tchanind(iix, iiy, iz) = 0
                                    end if
                                end do
                            end if
                            !
                            ! - get thickness and code crevasse:
                            !
                            thick = crt(ic, icrev) * (MAXCRX - real(jx)) / MAXCRX
                            nthick = 1 + int(thick / zsiz)
                            do iz = izhitem(ic), izhitem(ic) - nthick, -1
                                if(iz.ge.1) then
                                    tchannum(iix, iiy, iz) = ic
                                    tchanind(iix, iiy, iz) = 4
                                end if
                            end do
                            1                      continue
                        end if
                    end do
                end do
                !
                ! - finish crevasse templates:
                !
            end do

        end if
    end do
    !
    ! Calculate global proportion:
    !
    do i = 0, 4
        modprop(i) = 0.0
    end do
    do iz = 1, nz
        do iy = 1, ny
            do ix = 1, nx
                ii = tchanind(ix, iy, iz)
                modprop(ii) = modprop(ii) + 1.
            end do
        end do
    end do
    do i = 0, 4
        modprop(i) = modprop(i) / real(nx * ny * nz)
    end do
    !
    ! Calculate vertical proportion curves if needed:
    !
    if(lvert) then
        do iz = 1, nz
            do i = 1, 3
                pcurvea(iz, i) = 0.0
            end do
            do iy = 1, ny
                do ix = 1, nx
                    if(lvertng) then
                        if(tchannum(ix, iy, iz).ne.0)&
                                pcurvea(iz, 1) = pcurvea(iz, 1) + 1.
                    else
                        if(tchanind(ix, iy, iz).eq.1)&
                                pcurvea(iz, 1) = pcurvea(iz, 1) + 1.
                        if(tchanind(ix, iy, iz).eq.3)&
                                pcurvea(iz, 2) = pcurvea(iz, 2) + 1.
                        if(tchanind(ix, iy, iz).eq.4)&
                                pcurvea(iz, 3) = pcurvea(iz, 3) + 1.
                    end if
                end do
            end do
            do i = 1, 3
                pcurvea(iz, i) = pcurvea(iz, i) / real(nx * ny)
            end do
        end do
    end if
    !
    ! Calculate areal proportion maps if needed:
    !
    if(larea) then
        do iy = 1, ny
            do ix = 1, nx
                do i = 1, 3
                    pmapa(ix, iy, i) = 0.0
                end do
                do iz = 1, nz
                    if(lareang) then
                        if(tchannum(ix, iy, iz).ne.0)&
                                pmapa(ix, iy, 1) = pmapa(ix, iy, 1) + 1.
                    else
                        if(tchanind(ix, iy, iz).eq.1)&
                                pmapa(ix, iy, 1) = pmapa(ix, iy, 1) + 1.
                        if(tchanind(ix, iy, iz).eq.3)&
                                pmapa(ix, iy, 2) = pmapa(ix, iy, 2) + 1.
                        if(tchanind(ix, iy, iz).eq.4)&
                                pmapa(ix, iy, 3) = pmapa(ix, iy, 3) + 1.
                    end if
                end do
                do i = 1, 3
                    pmapa(ix, iy, i) = pmapa(ix, iy, i) / real(nz)
                end do
            end do
        end do
    end if
    !
    ! Calculate probability of each channel being chosen for perturbation:
    !
    do i = 1, mxc
        cprob(i) = 0.0
        if(chanon(i)) then
            cprob(i) = 1.0
            match(i) = 0.0
            missm(i) = 0.0
        end if
    end do
    do iwc = 1, nwd
        ix = xw(iwc)
        iy = yw(iwc)
        iz = zw(iwc)
        imod = tchannum(ix, iy, iz)
        if(fw(iwc).eq.0) then
            if(imod.ne.0) missm(imod) = missm(imod) + 1.
        else
            if(imod.ne.0) match(imod) = match(imod) + 1.
        end if
    end do
    do i = 1, mxc
        if(chanon(i)) then
            if((match(i) + missm(i)).gt.0.5) then
                fmiss = missm(i) / (match(i) + missm(i))
                if(fmiss.gt.0.3)&
                        cprob(i) = cprob(i) + (fmiss - 0.3) * 20.0
                if(fmiss.lt.0.3)&
                        cprob(i) = cprob(i) - (0.3 - fmiss) * 5.0
            end if
        end if
    end do
    !
    ! Return with raster image of channels:
    !
    return
end


subroutine getobj(obj, lreport)
    !-----------------------------------------------------------------------
    !
    ! Get the component objective functions for simulated annealing
    !
    !
    !
    !-----------------------------------------------------------------------
    use       geostat
    logical lreport
    !
    ! Matrix of objective function values for well:
    !
    real wobjfun(0:4, 0:4)
    data wobjfun/ 0.0, 1.0, 1.0, 0.5, 0.5, &
            1.0, 0.0, 0.0, 0.2, 0.2, &
            1.0, 0.0, 0.0, 0.2, 0.2, &
            0.5, 0.2, 0.2, 0.0, 0.2, &
            0.5, 0.2, 0.2, 0.2, 0.0  /
    !
    ! Establish total objective function:
    !
    obj = 0.0
    !
    ! Global proportions?
    !
    if(lglob) then

        obj = obj + sclglob * (tarprop(0) - modprop(0))&
                * (tarprop(0) - modprop(0))
        obj = obj + sclglob * (tarprop(1) - modprop(1))&
                * (tarprop(1) - modprop(1))
        if(ilev.eq.1) obj = obj + sclglob * (tarprop(3) - modprop(3))&
                * (tarprop(3) - modprop(3))
        if(icre.eq.1) obj = obj + sclglob * (tarprop(4) - modprop(4))&
                * (tarprop(4) - modprop(4))

        if(lreport) write(ldbg, 101) tarprop(0), modprop(0), &
                tarprop(1), modprop(1), &
                tarprop(3), modprop(3), &
                tarprop(4), modprop(4)
        101        format(/, ' Target shale    proportion = ', f6.4, &
                ' actual proportion = ', f6.4, /, &
                ' Target channel  proportion = ', f6.4, &
                ' actual proportion = ', f6.4, /, &
                ' Target levee    proportion = ', f6.4, &
                ' actual proportion = ', f6.4, /, &
                ' Target crevasse proportion = ', f6.4, &
                ' actual proportion = ', f6.4, /)
    end if
    !
    ! Vertical proportion curves?
    !
    if(lvert) then
        objt = 0.0
        do iz = 1, nz
            if(lvertng) then
                objt = objt + (pcurve(iz, 1) - pcurvea(iz, 1))&
                        * (pcurve(iz, 1) - pcurvea(iz, 1))
            else
                objt = objt + (pcurve(iz, 1) - pcurvea(iz, 1))&
                        * (pcurve(iz, 1) - pcurvea(iz, 1))
                if(ilev.eq.1)&
                        objt = objt + (pcurve(iz, 2) - pcurvea(iz, 2))&
                                * (pcurve(iz, 2) - pcurvea(iz, 2))
                if(icre.eq.1)&
                        objt = objt + (pcurve(iz, 3) - pcurvea(iz, 3))&
                                * (pcurve(iz, 3) - pcurvea(iz, 3))
            end if
        end do
        obj = obj + sclvert * objt
        !
        ! Report current values if requested:
        !
        if(lreport) then
            write(ldbg, 102)
            102           format(/, 'Vertical proportion curve reproduction')
            do iz = 1, nz
                write(ldbg, 103) iz, pcurve(iz, 1), pcurvea(iz, 1), &
                        pcurve(iz, 2), pcurvea(iz, 2), &
                        pcurve(iz, 3), pcurvea(iz, 3)
                103              format(i3, 1x, 3(1x, f7.4, 1x, f7.4))
            end do
        end if
    end if
    !
    ! Areal proportion maps?
    !
    if(larea) then
        objt = 0.0
        do iy = 1, ny
            do ix = 1, nx
                if(lareang) then
                    objt = objt + (pmap(ix, iy, 1) - pmapa(ix, iy, 1))&
                            * (pmap(ix, iy, 1) - pmapa(ix, iy, 1))
                else
                    objt = objt + (pmap(ix, iy, 1) - pmapa(ix, iy, 1))&
                            * (pmap(ix, iy, 1) - pmapa(ix, iy, 1))
                    if(ilev.eq.1)&
                            objt = objt + (pmap(ix, iy, 2) - pmapa(ix, iy, 2))&
                                    * (pmap(ix, iy, 2) - pmapa(ix, iy, 2))
                    if(icre.eq.1)&
                            objt = objt + (pmap(ix, iy, 3) - pmapa(ix, iy, 3))&
                                    * (pmap(ix, iy, 3) - pmapa(ix, iy, 3))
                end if
            end do
        end do
        obj = obj + sclarea * objt
        !
        ! Report current values if requested:
        !
        if(lreport) then
            write(ldbg, 105)
            105           format(/, 'Areal proportion map reproduction')
            do iy = 1, ny
                do ix = 1, nx
                    write(ldbg, 106) pmap(ix, iy, 1), pmapa(ix, iy, 1), &
                            pmap(ix, iy, 2), pmapa(ix, iy, 2), &
                            pmap(ix, iy, 3), pmapa(ix, iy, 3)
                    106                 format(3(1x, f7.4, 1x, f7.4))
                end do
            end do
        end if
    end if
    !
    ! Well data?
    !
    welltry = 0.
    if(lwell) then
        objt = 0.0
        do iwc = 1, nwd
            ix = xw(iwc)
            iy = yw(iwc)
            iz = zw(iwc)
            imod = tchanind(ix, iy, iz)
            iwel = fw(iwc)
            objt = objt + wobjfun(imod, iwel) * wfac
            if(iwel.ne.imod) welltry = welltry + 1.
        end do
        welltry = welltry / real(max(nwd, 1)) * 100.0
        obj = obj + sclwell * objt
    end if
    !
    ! Finished with objective function:
    !
    return
end


subroutine getpert
    !-----------------------------------------------------------------------
    !
    ! Specify a perturbation for conditioning
    !
    ! Probabilities:  1.  cumprob(1)               --> one on and one off
    !                 2.  cumprob(2) - cumprob(1)  --> one on
    !                 3.  cumprob(3) - cumprob(2)  --> one off
    !                 4.     1.0     - cumprob(3)  --> one on and one off
    !
    ! Handle wells in beginning
    !
    !
    !-----------------------------------------------------------------------
    use       geostat
    real*8    acorni
    logical   flong
    !
    ! ACORN parameters:
    !
    parameter(KORDEI = 12, MAXOP1 = KORDEI + 1, MAXINT = 2**30)
    common /iaco/ ixv(MAXOP1)
    !
    ! Initialize counters of changes made this perturbation:
    !
    flong = .false.
    nonp = 0
    noffp = 0
    nvshift = 0
    !
    ! Find a place to turn a channel on (don't want to overwrite a channel
    ! that has been turned off and that may need to be turned on again).
    !
    ion = -1
    do i = 1, mxc + 1
        if(.not.chanon(i).and.ion.le.0) ion = i
    end do
    if(ion.le.0.or.ion.gt.MAXC) ion = 1
    icon(1) = ion
    !
    ! Decide what to do:
    !
    10   pval = real(acorni(idum))
    if(pval.le.cumprob(1)) go to 1
    if(pval.le.cumprob(2)) go to 2
    if(pval.le.cumprob(3)) go to 3
    !
    ! Option 1: turn a channel off and turn another on:
    !
    1    continue
    call getchoff(i)
    noffp = noffp + 1
    icoff(noffp) = i
    chanon(i) = .false.
    nonp = 1
    ion = icon(1)
    call getchpar(ion, flong)
    return
    !
    ! Option 2: turn a channel on:
    !
    2    continue
    nonp = 1
    ion = icon(1)
    call getchpar(ion, flong)
    return
    !
    ! Option 3: turn a channel off:
    !
    3    continue
    call getchoff(i)
    noffp = noffp + 1
    icoff(noffp) = i
    chanon(i) = .false.
    !
    ! Return to fluvsim to raster, calculate objective function, ...
    !
    return
end


subroutine getchw
    !-----------------------------------------------------------------------
    !
    ! Loop over all well intervals and add channels that honor the channel
    ! intersections at well locations.
    !
    !
    !
    !-----------------------------------------------------------------------
    use       geostat
    real      fctt(3)
    real*8    acorni
    logical   finint, flong
    !
    ! ACORN parameters:
    !
    parameter(KORDEI = 12, MAXOP1 = KORDEI + 1, MAXINT = 2**30)
    common /iaco/ ixv(MAXOP1)
    !
    ! Loop over all well intervals:
    !
    if(nwint.le.0) return
    do iwint = 1, nwint
        !
        ! Only deal with channel sand intersections at this time:
        !
        if(facint(iwint).ne.1) go to 1
        !
        ! Check the mismatch (some other channel going through a well may cover
        ! this intersection):
        !
        nsize = ninwint(iwint)
        nmm = 0
        iztop = 0
        izbot = 999
        do i = 1, nsize
            iix = ixint(iwint, i)
            iiy = iyint(iwint, i)
            iiz = izint(iwint, i)
            if(iiz.gt.iztop) iztop = iiz
            if(iiz.lt.izbot) izbot = iiz
            rarray(i) = abs(tchannum(iix, iiy, iiz))
            if(tchanind(iix, iiy, iiz).ne.1) nmm = nmm + 1
        end do
        if(nmm.eq.0) go to 1
        !VD July 30 1998
        xxw = xmn + real(iix - 1.0) * xsiz
        yyw = ymn + real(iiy - 1.0) * ysiz
        !
        ! Add channels until this intersection is "covered":
        !
        ztop = zmn + real(iztop - 0.5) * zsiz
        zbot = zmn + real(izbot - 1.5) * zsiz
        zcur = ztop
        if(zbot.ge.ztop) go to 1
        finint = .false.
        2          continue
        !
        ! Draw an apparent thickness, determine top of channel (could be above
        ! current top but not above top of intersection), check bottom of
        ! interval, and see if this is last channel needed for this interval:
        !
        call drawat(athk)
        zch = zcur
        if(zcur.lt.ztop) then
            zch = zch + real(acorni(idum)) * 0.5 * athk
            if(zch.gt.ztop) zch = ztop
        end if
        zchb = zch - athk
        if(zchb.le.zbot) then
            athk = zch - zbot
            finint = .true.
        end if
        !
        ! Find a place to add a channel:
        !
        ion = -1
        do i = 1, mxc + 1
            if(.not.chanon(i).and.ion.le.0) ion = i
        end do
        if(ion.le.0.or.ion.gt.MAXC) ion = 1
        !
        ! Get maximum thickness greater than apparent thickness:
        !
        ntry = 0
        4          thk = getval(fct)
        ntry = ntry + 1
        if(ntry.gt.100) thk = 1.1 * athk
        if(thk.le.athk) go to 4
        !
        ! Add channel with specified thickness:
        !
        ngetc = 0
        3          ngetc = ngetc + 1
        if(ngetc.gt.10) go to 1
        fctt(1) = fct(1)
        fctt(2) = fct(2)
        fctt(3) = fct(3)
        fct(1) = thk
        fct(2) = thk
        fct(3) = thk
        flong = .true.
        call getchpar(ion, flong)
        flong = .false.
        fct(1) = fctt(1)
        fct(2) = fctt(2)
        fct(3) = fctt(3)
        !
        ! Shift the channel to the correct vertical position:
        !
        ndelz = int(zch / zsiz + 0.5) - izhitem(ion)
        call vshift(ion, ndelz)
        !
        ! Shift the channel laterally:
        !
        if(abs(csina(ion)).gt.abs(ccosa(ion))) then
            !
            ! Shift "Y" origin up or down to intersect well at correct apparent
            ! thickness.  First, randomly decide whether to shift from the bottom
            ! or the top and then loop over many different possible Y origins:
            !
            if(real(acorni(idum)).lt.0.5) then
                iys = -int(cymax - cymin) / cdy
                iye = int(cymax - cymin) / cdy
                iyinc = 1
            else
                iys = int(cymax - cymin) / cdy
                iye = -int(cymax - cymin) / cdy
                iyinc = -1
            end if
            xorig = cxorigin(ion)
            do indy = iys, iye, iyinc
                yorig = cymin + (real(indy) - 0.5) * cdy
                !
                ! Now, see if the channel is at the well location with (at least) the
                ! apparent thickness:
                !
                xchan = ccosa(ion) * (xxw - xorig)&
                        - csina(ion) * (yyw - yorig)
                ychan = csina(ion) * (xxw - xorig)&
                        + ccosa(ion) * (yyw - yorig)
                iyc = int(ychan / cdy + 1.5)
                if(xchan.ge.0) then
                    ixc = int(xchan / cdy + 0.5)
                else
                    ixc = int(xchan / cdy - 0.5)
                end if
                if(iyc.ge.1.and.iyc.le.MAXL) then
                    if(ixc.ge.ixlotem(ion, iyc).and.&
                            ixc.le.ixhitem(ion, iyc)) then
                        cthik = real(1 + izhitem(ion)&
                                - izlotem(ion, iyc, ixc))&
                                * zsiz
                        if(cthik.ge.athk) then
                            cyorigin(ion) = yorig
                            go to 601
                        end if
                    end if
                end if
            end do
            !
            ! Could not shift to match well - must be too small or something - go
            ! back and get another channel.
            !
            go to 3
            601        continue
        else
            !
            ! Shift "X" origin right or left to intersect well at correct apparent
            ! thickness.  First, randomly decide whether to shift from the right
            ! or the left and then loop over many different possible X origins:
            !
            if(real(acorni(idum)).lt.0.5) then
                ixs = -int(cxmax - cxmin) / cdy
                ixe = int(cxmax - cxmin) / cdy
                ixinc = 1
            else
                ixs = int(cxmax - cxmin) / cdy
                ixe = -int(cxmax - cxmin) / cdy
                ixinc = -1
            end if
            yorig = cyorigin(ion)
            do indx = ixs, ixe, ixinc
                xorig = cxmin + (real(indx) - 0.5) * cdy
                !
                ! Now, see if the channel is at the well location with (at least) the
                ! apparent thickness:
                !
                xchan = ccosa(ion) * (xxw - xorig)&
                        - csina(ion) * (yyw - yorig)
                ychan = csina(ion) * (xxw - xorig)&
                        + ccosa(ion) * (yyw - yorig)
                iyc = int(ychan / cdy + 1.5)
                if(xchan.ge.0) then
                    ixc = int(xchan / cdy + 0.5)
                else
                    ixc = int(xchan / cdy - 0.5)
                end if
                if(iyc.ge.1.and.iyc.le.MAXL) then
                    if(ixc.ge.ixlotem(ion, iyc).and.&
                            ixc.le.ixhitem(ion, iyc)) then
                        cthik = real(1 + izhitem(ion)&
                                - izlotem(ion, iyc, ixc))&
                                * zsiz
                        if(cthik.ge.athk) then
                            cxorigin(ion) = xorig
                            go to 602
                        end if
                    end if
                end if
            end do
            !
            ! Could not shift to match well - must be too small or something - go
            ! back and get another channel.
            !
            go to 3
            602        continue
        end if
        !
        ! Are we done with this interval?
        !
        zcur = zcur - cthik
        if(finint.or.zcur.le.(zbot + 0.5 * zsiz)) go to 1
        go to 2
        1          continue
        !
        ! End loop over all channel intersections:
        !
    end do
    !
    ! Return to fluvsim to add the rest of the starting channels:
    !
    return
end


subroutine vshift(ic, ndelz)
    !-----------------------------------------------------------------------
    !
    ! Shift channel "ic" by ndelz grid units, i.e., reset all arrays for
    ! the channel.
    !
    !
    !-----------------------------------------------------------------------
    use       geostat
    !
    ! Shift, but make sure the integer pointers stay in the model:
    !
    cz(ic) = cz(ic) + real(ndelz) * zsiz
    izhitem(ic) = izhitem(ic) + ndelz
    izhitem(ic) = min(max(1, izhitem(ic)), nz)
    do iy = 1, nyc(ic)
        rlzlo(ic, iy) = rlzlo(ic, iy) + ndelz
        rlzlo(ic, iy) = min(max(1, rlzlo(ic, iy)), nz)
        llzlo(ic, iy) = llzlo(ic, iy) + ndelz
        llzlo(ic, iy) = min(max(1, llzlo(ic, iy)), nz)
        do ix = ixlotem(ic, iy), ixhitem(ic, iy)
            izlotem(ic, iy, ix) = izlotem(ic, iy, ix) + ndelz
            izlotem(ic, iy, ix) = min(max(1, izlotem(ic, iy, ix)), nz)
        end do
    end do
    !
    ! All finished:
    !
    return
end


real function getval(fdist)
    !-----------------------------------------------------------------------
    !
    !
    !
    !-----------------------------------------------------------------------
    real      fdist(3)
    real*8    acorni
    !
    ! ACORN parameters:
    !
    parameter(KORDEI = 12, MAXOP1 = KORDEI + 1, MAXINT = 2**30)
    common /iaco/ ixv(MAXOP1)
    !
    ! Draw a value from triangular distribution:
    !
    cdf = real(acorni(idum))
    if(cdf.lt.0.5) then
        getval = fdist(1) + 2.0 * cdf * (fdist(2) - fdist(1))
    else
        getval = fdist(2) + 2.0 * (cdf - 0.5) * (fdist(3) - fdist(2))
    end if

    return
end


subroutine csect(xleft, xright, relpos, tmax, xpos, thick)
    !-----------------------------------------------------------------------
    !
    !
    !
    !
    !
    !-----------------------------------------------------------------------
    parameter(EPSLON = 0.0001)
    width = xright - xleft
    xx = xpos - xleft
    if(width.le.EPSLON.or.relpos.le.EPSLON&
            .or.(1. - relpos).le.EPSLON.or.xx.le.0.0.or.xx.ge.width) then
        thick = 0.0
        return
    end if
    if(relpos.le.0.5) then
        b = -log(2.) / log(relpos)
        thick = 4. * tmax * ((xx / width)**b) * ((1. - (xx / width)**b))
    else
        b = -log(2.) / log(1. - relpos)
        thick = 4. * tmax * ((1. - xx / width)**b) * (1. - ((1. - xx / width)**b))
    end if
    return
end


subroutine get1d(ny, ysiz, range, array)
    !-----------------------------------------------------------------------
    !
    !            Stochastic Averaging Simulation for 1-D String
    !            **********************************************
    !
    !
    !
    !
    !-----------------------------------------------------------------------
    !
    ! Variable Declaration:
    !
    parameter(MAXY = 750, MAXLEN = 250)
    real      tarray(MAXY + 2 * MAXLEN), array(*)
    real*8    p, acorni
    !
    ! ACORN parameters:
    !
    parameter(KORDEI = 12, MAXOP1 = KORDEI + 1, MAXINT = 2**30)
    common /iaco/ ixv(MAXOP1)
    !
    ! Do we have enough storage?
    !
    if(ny.gt.MAXY) then
        write(*, *) ' requested too large simulation in get1d ', ny
        write(*, *) ' available size is ', MAXY
        stop
    end if
    !
    ! Size of smoothing window:
    !
    n1 = min(int((range * 0.10) / ysiz + 0.5), MAXLEN)
    n2 = min(int((range * 0.50) / ysiz + 0.5), MAXLEN)
    !
    ! Generate ny Gaussian random numbers:
    !
    do i = 1, ny + 2 * n2
        1          p = acorni(idum)
        call gauinv(p, xp, ierr)
        if(xp.lt.(-6.).or.xp.gt.(6.)) go to 1
        tarray(i) = xp
    end do
    !
    ! Compute the simulation at each grid point:
    !
    do i = 1, ny
        loc = i + n2
        value = 0.0
        llo = loc - n2
        lhi = loc - n1
        do j = llo, lhi
            value = value + tarray(j) * real((j - llo + 1)) / &
                    real((lhi - llo + 1))
        end do
        llo = lhi + 1
        lhi = loc + n1 - 1
        do j = llo, lhi
            value = value + tarray(j)
        end do
        llo = lhi + 1
        lhi = loc + n2
        do j = llo, lhi
            value = value + tarray(j) * real((lhi - j + 1)) / &
                    real((lhi - llo + 1))
        end do
        array(i) = value
    end do
    !
    ! A smoothing pass:
    !
    tarray(1) = 3.0 * array(1)
    do i = 2, ny - 1
        tarray(i) = array(i - 1) + array(i) + array(i + 1)
    end do
    tarray(ny) = 3.0 * array(ny)
    !
    ! Restandardize the results to mean 0 and variance 1
    !
    xmean = 0.0
    xvar = 0.0
    do i = 1, ny
        xmean = xmean + tarray(i)
        xvar = xvar + tarray(i) * tarray(i)
    end do
    xmean = xmean / real(ny)
    xvar = xvar / real(ny) - xmean * xmean
    xstd = sqrt(xvar)
    do i = 1, ny
        array(i) = (tarray(i) - xmean) / xstd
    end do
    !
    ! Return with this realization:
    !
    return
end


subroutine getcre
    !-----------------------------------------------------------------------
    !
    !
    !
    !
    !-----------------------------------------------------------------------
    use       geostat
    real*8    acorni
    !
    ! ACORN parameters:
    !
    parameter(KORDEI = 12, MAXOP1 = KORDEI + 1, MAXINT = 2**30)
    common /iaco/ ixv(MAXOP1)
    !
    ! Loop over all crevasses:
    !
    do ic = 1, MAXCRE
        do ix = 0, MAXCRX
            do iy = -MAXCRY, MAXCRY
                cre(ic, ix, iy) = .false.
            end do
        end do
        !
        ! Draw values for the crevasse parameters:
        !
        lenxc = 1 + int(getval(fcrlen))
        lencr = 1 + int(getval(fcrwl))
        nwalk = 1 + int(getval(fcrnw))
        dlat = getval(fcrlat)
        if(lenxc.gt.(MAXCRY / 2)) lenxc = MAXCRY / 2
        !
        ! Send out the random walkers and get the crevasse shape:
        !
        dlatlo = 0.5 * dlat
        dlathi = 1.0 - 0.5 * dlat
        do jy = -lenxc, lenxc
            do iwalk = 1, nwalk
                ix = 0
                iy = jy
                cre(ic, ix, iy) = .true.
                do ilen = 1, lencr
                    rtmp = real(acorni(idum))
                    if(rtmp.le.dlatlo) then
                        iy = iy - 1
                        if(iy.lt.-MAXCRY) iy = -MAXCRY
                    else if(rtmp.ge.dlathi) then
                        iy = iy + 1
                        if(iy.gt. MAXCRY) iy = MAXCRY
                    else
                        ix = ix + 1
                        if(ix.gt. MAXCRX) ix = MAXCRX
                    end if
                    cre(ic, ix, iy) = .true.
                end do
            end do
        end do
        !
        ! End loop over all crevasses:
        !
    end do
    !
    ! Finished:
    !
    return
end


subroutine getat
    !-----------------------------------------------------------------------
    !
    !             Calculate Apparent Thickness Distribution
    !             *****************************************
    !
    ! Determine the apparent thickness distribution corresponding to the
    ! initial guess of maximum thickness.
    !
    !
    !
    ! C. Deutsch / T. Tran                                        March 1998
    !-----------------------------------------------------------------------
    use       geostat
    parameter(NCLS = 10)
    !
    ! Establish the thickness thresholds:
    !
    tinc = fct(3) / real(MAXDIS - 1)
    atcut(1) = 0.5 * tinc
    do i = 2, MAXDIS - 1
        atcut(i) = atcut(i - 1) + tinc
    end do
    nthres = MAXDIS - 1
    !
    ! Initialize cdf values:
    !
    do i = 1, MAXDIS
        atcdf(i) = 0.0
    end do
    tpdf = 0.0
    !
    ! FIRST Loop over range of allowable maximum thickness:
    !
    tinc = (fct(3) - fct(1)) / real(NCLS)
    thick = fct(1) - tinc / 2.0
    do it = 1, NCLS
        thick = thick + tinc
        !
        !     get probability of this thickness:
        !
        fmax = 2.0 / (fct(3) - fct(1))
        if(thick.le.fct(2)) then
            fprob = (thick - fct(1)) / (fct(2) - fct(1)) * fmax
        else
            fprob = (fct(3) - thick) / (fct(3) - fct(2)) * fmax
        end if
        !
        ! SECOND Loop over range of allowable width / thickness ratios:
        !
        winc = (fcwt(3) - fcwt(1)) / real(NCLS)
        width = fcwt(1) - winc / 2.0
        do iw = 1, NCLS
            width = width + winc
            !
            !     get probability of this width:
            !
            gmax = 2.0 / (fcwt(3) - fcwt(1))
            if(width.le.fcwt(2)) then
                gprob = (width - fcwt(1)) / (fcwt(2) - fcwt(1)) * gmax
            else
                gprob = (fcwt(3) - width) / (fcwt(3) - fcwt(2)) * gmax
            end if
            !
            ! THIRD Loop over range of allowable relative positions:
            !
            pinc = 1.0 / real(NCLS)
            pos = -pinc / 2.0
            do ip = 1, NCLS
                pos = pos + pinc
                !
                ! FOURTH Loop along width:
                !
                ainc = width / real(NCLS)
                dis = -ainc / 2.0
                do ia = 1, NCLS
                    dis = dis + ainc
                    !
                    ! Now, calculate the thickness:
                    !
                    call csect(0.0, width, pos, thick, dis, thktmp)
                    icls = 1
                    do i = 1, nthres
                        if(thktmp.ge.atcut(i)) icls = i + 1
                    end do
                    atcdf(icls) = atcdf(icls) + fprob * gprob
                    tpdf = tpdf + fprob * gprob
                    !
                    ! END four loop over channel sizes ...
                    !
                end do
            end do
        end do
    end do
    !
    ! Turn apparent thicknesses into a cdf:
    !
    tpdf = 1.0 / tpdf
    oldcp = 0.0
    cp = 0.0
    do i = 1, nthres
        cp = cp + atcdf(i) * tpdf
        atcdf(i) = (cp + oldcp) * 0.5
        oldcp = cp
    end do
    !
    ! All finished:
    !
    return
end


subroutine drawat(athk)
    !-----------------------------------------------------------------------
    !
    !                    Draw an Apparent Thickness
    !                    **************************
    !
    ! Draw an apparent thickness from the distribution established earlier.
    !
    !
    !
    ! C. Deutsch / T. Tran                                        March 1998
    !-----------------------------------------------------------------------
    use       geostat
    real*8    p, acorni
    !
    ! ACORN parameters:
    !
    parameter(KORDEI = 12, MAXOP1 = KORDEI + 1, MAXINT = 2**30)
    common /iaco/ ixv(MAXOP1)
    !
    ! Determine the cdf and range (lower, middle, upper):
    !
    nthres = MAXDIS - 1
    acdf = real(acorni(idum))
    if(acdf.le.atcdf(1)) then
        athk = atcut(1) * acdf / atcdf(1)
    else if(acdf.ge.atcdf(nthres)) then
        athk = atcut(nthres) + (fct(3) - atcut(nthres)) * &
                (acdf - atcdf(nthres)) / (1.0 - atcdf(nthres))
    else
        do i = 2, nthres
            if(acdf.le.atcdf(i)) then
                athk = atcut(i - 1) + (atcut(i) - atcut(i - 1)) * &
                        (acdf - atcdf(i - 1)) / (atcdf(i) - atcdf(i - 1))
                return
            end if
        end do
    end if
    !
    ! Return with apparent thickness:
    !
    return
end


subroutine getchoff(ioff)
    !-----------------------------------------------------------------------
    !
    !                    Draw a Channel to Turn Off
    !                    **************************
    !
    ! Use the "cprop" array as probabilities to pick a channel to turn off.
    !
    !
    !
    ! C. Deutsch / T. Tran                                        April 1998
    !-----------------------------------------------------------------------
    use       geostat
    real*8    p, acorni
    !
    ! ACORN parameters:
    !
    parameter(KORDEI = 12, MAXOP1 = KORDEI + 1, MAXINT = 2**30)
    common /iaco/ ixv(MAXOP1)
    !
    ! Copy the probabilities into "missm" and use it for drawing:
    !
    ioff = 1
    sumwts = 0.0
    do i = 1, mxc
        missm(i) = cprob(i)
        if(missm(i).lt.0.0) missm(i) = 0.0
        sumwts = sumwts + missm(i)
    end do
    if(sumwts.le.0) return
    !
    ! Turn into a CDF:
    !
    oldcp = 0.0
    cp = 0.0
    sumwts = 1.0 / sumwts
    do i = 1, mxc
        missm(i) = oldcp + missm(i) * sumwts
        oldcp = missm(i)
    end do
    !
    ! Draw from cdf:
    !
    cdf = real(acorni(idum))
    if(cdf.le.missm(1)) then
        ioff = 1
        go to 1
    end if
    do i = 2, mxc
        if(cdf.le.missm(i).and.cdf.ge.missm(i - 1)) then
            ioff = i
            go to 1
        end if
    end do
    !
    ! Return with channel to turn off:
    !
    1    continue
    return
end


subroutine makepar
    !-----------------------------------------------------------------------
    !
    !                      Write a Parameter File
    !                      **********************
    !
    !
    !
    !-----------------------------------------------------------------------
    lun = 99
    open(lun, file = 'fluvsim.par', status = 'UNKNOWN')
    write(lun, 10)
    10   format('                  Parameters for FLUVSIM', /, &
            '                  **********************', /, /, &
            'START OF PARAMETERS:')
    write(lun, 11)
    11   format('data/well01.dat                ', &
            '-file with well conditioning data')
    write(lun, 12)
    12   format('1  2  3  4  5                  ', &
            '-  columns for X, Y, Z, well #, facies')
    write(lun, 13)
    13   format('-1.0       1.0e21              ', &
            '-  trimming limits')
    write(lun, 14)
    14   format('1                              ', &
            '-debugging level: 0,1,2,3')
    write(lun, 15)
    15   format('output/fluvsim.dbg             ', &
            '-file for debugging output')
    write(lun, 16)
    16   format('output/fluvsim.geo             ', &
            '-file for geometric specification')
    write(lun, 17)
    17   format('output/fluvsim.out             ', &
            '-file for simulation output')
    write(lun, 18)
    18   format('output/fluvsim.vp              ', &
            '-file for vertical prop curve output')
    write(lun, 19)
    19   format('output/fluvsim.ap              ', &
            '-file for areal prop map output')
    write(lun, 20)
    20   format('output/fluvsim.wd              ', &
            '-file for well data output')
    write(lun, 21)
    21   format('1                              ', &
            '-number of realizations to generate')
    write(lun, 22)
    22   format('100   0.0   40.0               ', &
            '-nx,xmn,xsiz - geological coordinates')
    write(lun, 23)
    23   format('100   0.0   40.0               ', &
            '-ny,ymn,ysiz - geological coordinates')
    write(lun, 24)
    24   format('50          50.0               ', &
            '-nz, average thickness in physical units')
    write(lun, 25)
    25   format('69069                          ', &
            '-random number seed')
    write(lun, 26)
    26   format('1   0   0   1                  ', &
            '-1=on,0=off: global, vert, areal, wells')
    write(lun, 27)
    27   format('1.  1.  1.  1.                 ', &
            '-weighting : global, vert, areal, wells')
    write(lun, 28)
    28   format('100   10   0.05                ', &
            '-maximum iter, max no change, min. obj.')
    write(lun, 29)
    29   format('0.0   0.10   3  1  8           ', &
            '-annealing schedule: t0,redfac,ka,k,num')
    write(lun, 30)
    30   format('0.1 0.1 0.1 1.0                ', &
            '-Pert prob: 1on+1off, 1on, 1off, fix well')
    write(lun, 31)
    31   format('   1    0    0                 ', &
            '-Facies(on): channel, levee, crevasse')
    write(lun, 32)
    32   format('0.30 0.10 0.10                 ', &
            '-Proportion: channel, levee, crevasse')
    write(lun, 33)
    33   format('pcurve.dat                     ', &
            '-  vertical proportion curves')
    write(lun, 34)
    34   format('0                              ', &
            '-     0=net-to-gross, 1=all facies')
    write(lun, 35)
    35   format('1  7  8                        ', &
            '-     column numbers')
    write(lun, 36)
    36   format('arealprop.dat                  ', &
            '-  areal proportion map')
    write(lun, 37)
    37   format('1                              ', &
            '-     0=net-to-gross, 1=all facies')
    write(lun, 38)
    38   format('2  3  4                        ', &
            '-     column numbers')
    write(lun, 39)
    39   format('150                            ', &
            '-maximum number of channels')
    write(lun, 40)
    40   format('-30.0    0.0    30.0           ', &
            '-channel:  orientation (degrees)')
    write(lun, 41)
    41   format('200.0  200.0   200.0           ', &
            '-channel:  sinuosity: average departure')
    write(lun, 42)
    42   format('800.0  800.0   800.0           ', &
            '-channel:  sinuosity: length scale')
    write(lun, 43)
    43   format('  1.0    3.0     5.0           ', &
            '-channel:  thickness')
    write(lun, 44)
    44   format('  1.0    1.0     1.0           ', &
            '-channel:  thickness undulation')
    write(lun, 45)
    45   format('250.0  400.0   450.0           ', &
            '-channel:  thickness undul. length scale')
    write(lun, 46)
    46   format('150.0  200.0   250.0           ', &
            '-channel:  width/thickness ratio')
    write(lun, 47)
    47   format('  1.0    1.0     1.0           ', &
            '-channel:  width: undulation')
    write(lun, 48)
    48   format('250.0  250.0   250.0           ', &
            '-channel:  width: undulation length scale')
    write(lun, 49)
    49   format('160.0  240.0   320.0           ', &
            '-levee:    average width')
    write(lun, 50)
    50   format('  0.1    0.1     0.1           ', &
            '-levee:    average height')
    write(lun, 51)
    51   format('  0.2    0.3     0.4           ', &
            '-levee:    depth below top')
    write(lun, 52)
    52   format(' 80.0   80.0    80.0           ', &
            '-crevasse: attachment length')
    write(lun, 53)
    53   format('  0.25   0.5     0.75          ', &
            '-crevasse: relative thickness by channel')
    write(lun, 54)
    54   format('500.0  500.0   500.0           ', &
            '-crevasse: areal size (diameter)')

    close(lun)
    return
end


double precision function acorni(idum)
    !-----------------------------------------------------------------------
    !
    ! Fortran implementation of ACORN random number generator of order less
    ! than or equal to 12 (higher orders can be obtained by increasing the
    ! parameter value MAXORD).
    !
    !
    ! NOTES: 1. The variable idum is a dummy variable. The common block
    !           IACO is used to transfer data into the function.
    !
    !        2. Before the first call to ACORN the common block IACO must
    !           be initialised by the user, as follows. The values of
    !           variables in the common block must not subsequently be
    !           changed by the user.
    !
    !             KORDEI - order of generator required ( must be =< MAXORD)
    !
    !             MAXINT - modulus for generator, must be chosen small
    !                      enough that 2*MAXINT does not overflow
    !
    !             ixv(1) - seed for random number generator
    !                      require 0 < ixv(1) < MAXINT
    !
    !             (ixv(I+1),I=1,KORDEI)
    !                    - KORDEI initial values for generator
    !                      require 0 =< ixv(I+1) < MAXINT
    !
    !        3. After initialisation, each call to ACORN generates a single
    !           random number between 0 and 1.
    !
    !        4. An example of suitable values for parameters is
    !
    !             KORDEI   = 10
    !             MAXINT   = 2**30
    !             ixv(1)   = an odd integer in the (approximate) range
    !                        (0.001 * MAXINT) to (0.999 * MAXINT)
    !             ixv(I+1) = 0, I=1,KORDEI
    !
    !
    !
    ! Author: R.S.Wikramaratna,                           Date: October 1990
    !-----------------------------------------------------------------------
    implicit double precision (a-h, o-z)
    parameter (KORDEI = 12, MAXOP1 = KORDEI + 1, MAXINT = 2**30)
    common/iaco/ ixv(MAXOP1)
    do i = 1, KORDEI
        ixv(i + 1) = (ixv(i + 1) + ixv(i))
        if(ixv(i + 1).ge.MAXINT) ixv(i + 1) = ixv(i + 1) - MAXINT
    end do
    acorni = dble(ixv(KORDEI + 1)) / MAXINT
    return
end


subroutine chknam(str, len)
    !-----------------------------------------------------------------------
    !
    !                   Check for a Valid File Name
    !                   ***************************
    !
    ! This subroutine takes the character string "str" of length "len" and
    ! removes all leading blanks and blanks out all characters after the
    ! first blank found in the string (leading blanks are removed first).
    !
    !
    !
    !-----------------------------------------------------------------------
    parameter (MAXLEN = 132)
    character str(MAXLEN)*1
    !
    ! Remove leading blanks:
    !
    do i = 1, len - 1
        if(str(i).ne.' ') then
            if(i.eq.1) go to 1
            do j = 1, len - i + 1
                k = j + i - 1
                str(j) = str(k)
            end do
            do j = len, len - i + 2, -1
                str(j) = ' '
            end do
            go to 1
        end if
    end do
    1    continue
    !
    ! Find first blank and blank out the remaining characters:
    !
    do i = 1, len - 1
        if(str(i).eq.' ') then
            do j = i + 1, len
                str(j) = ' '
            end do
            go to 2
        end if
    end do
    2    continue
    !
    ! Return with modified file name:
    !
    return
end


subroutine gauinv(p, xp, ierr)
    !-----------------------------------------------------------------------
    !
    ! Computes the inverse of the standard normal cumulative distribution
    ! function with a numerical approximation from : Statistical Computing,
    ! by W.J. Kennedy, Jr. and James E. Gentle, 1980, p. 95.
    !
    !
    !
    ! INPUT/OUTPUT:
    !
    !   p    = double precision cumulative probability value: dble(psingle)
    !   xp   = G^-1 (p) in single precision
    !   ierr = 1 - then error situation (p out of range), 0 - OK
    !
    !
    !-----------------------------------------------------------------------
    real*8 p0, p1, p2, p3, p4, q0, q1, q2, q3, q4, y, pp, lim, p
    save   p0, p1, p2, p3, p4, q0, q1, q2, q3, q4, lim
    !
    ! Coefficients of approximation:
    !
    data lim/1.0e-10/
    data p0/-0.322232431088/, p1/-1.0/, p2/-0.342242088547/, &
            p3/-0.0204231210245/, p4/-0.0000453642210148/
    data q0/0.0993484626060/, q1/0.588581570495/, q2/0.531103462366/, &
            q3/0.103537752850/, q4/0.0038560700634/
    !
    ! Check for an error situation:
    !
    ierr = 1
    if(p.lt.lim) then
        xp = -1.0e10
        return
    end if
    if(p.gt.(1.0 - lim)) then
        xp = 1.0e10
        return
    end if
    ierr = 0
    !
    ! Get k for an error situation:
    !
    pp = p
    if(p.gt.0.5) pp = 1 - pp
    xp = 0.0
    if(p.eq.0.5) return
    !
    ! Approximate the function:
    !
    y = dsqrt(dlog(1.0 / (pp * pp)))
    xp = real(y + ((((y * p4 + p3) * y + p2) * y + p1) * y + p0) / &
            ((((y * q4 + q3) * y + q2) * y + q1) * y + q0))
    if(real(p).eq.real(pp)) xp = -xp
    !
    ! Return with G^-1(p):
    !
    return
end


subroutine getindx(n, min, siz, loc, index, inflag)
    !-----------------------------------------------------------------------
    !
    !     Gets the coordinate index location of a point within a grid
    !     ***********************************************************
    !
    !
    ! n       number of "nodes" or "cells" in this coordinate direction
    ! min     origin at the center of the first cell
    ! siz     size of the cells
    ! loc     location of the point being considered
    ! index   output index within [1,n]
    ! inflag  true if the location is actually in the grid (false otherwise
    !         e.g., if the location is outside then index will be set to
    !         nearest boundary
    !
    !
    !
    !-----------------------------------------------------------------------
    integer   n, index
    real      min, siz, loc
    logical   inflag
    !
    ! Compute the index of "loc":
    !
    index = int((loc - min) / siz + 1.5)
    !
    ! Check to see if in or out:
    !
    if(index.lt.1) then
        index = 1
        inflag = .false.
    else if(index.gt.n) then
        index = n
        inflag = .false.
    else
        inflag = .true.
    end if
    !
    ! Return to calling program:
    !
    return
end


subroutine sortem(ib, ie, a, iperm, b, c, d, e, f, g, h)
    !-----------------------------------------------------------------------
    !
    !                      Quickersort Subroutine
    !                      **********************
    !
    ! This is a subroutine for sorting a real array in ascending order. This
    ! is a Fortran translation of algorithm 271, quickersort, by R.S. Scowen
    ! in collected algorithms of the ACM.
    !
    ! The method used is that of continually splitting the array into parts
    ! such that all elements of one part are less than all elements of the
    ! other, with a third part in the middle consisting of one element.  An
    ! element with value t is chosen arbitrarily (here we choose the middle
    ! element). i and j give the lower and upper limits of the segment being
    ! split.  After the split a value q will have been found such that
    ! a(q)=t and a(l)<=t<=a(m) for all i<=l<q<m<=j.  The program then
    ! performs operations on the two segments (i,q-1) and (q+1,j) as follows
    ! The smaller segment is split and the position of the larger segment is
    ! stored in the lt and ut arrays.  If the segment to be split contains
    ! two or fewer elements, it is sorted and another segment is obtained
    ! from the lt and ut arrays.  When no more segments remain, the array
    ! is completely sorted.
    !
    !
    ! INPUT PARAMETERS:
    !
    !   ib,ie        start and end index of the array to be sorteda
    !   a            array, a portion of which has to be sorted.
    !   iperm        0 no other array is permuted.
    !                1 array b is permuted according to array a
    !                2 arrays b,c are permuted.
    !                3 arrays b,c,d are permuted.
    !                4 arrays b,c,d,e are permuted.
    !                5 arrays b,c,d,e,f are permuted.
    !                6 arrays b,c,d,e,f,g are permuted.
    !                7 arrays b,c,d,e,f,g,h are permuted.
    !               >7 no other array is permuted.
    !
    !   b,c,d,e,f,g,h  arrays to be permuted according to array a.
    !
    ! OUTPUT PARAMETERS:
    !
    !    a      = the array, a portion of which has been sorted.
    !
    !    b,c,d,e,f,g,h  =arrays permuted according to array a (see iperm)
    !
    ! NO EXTERNAL ROUTINES REQUIRED:
    !
    !-----------------------------------------------------------------------
    dimension a(*), b(*), c(*), d(*), e(*), f(*), g(*), h(*)
    !
    ! The dimensions for lt and ut have to be at least log (base 2) n
    !
    integer   lt(64), ut(64), i, j, k, m, p, q
    !
    ! Initialize:
    !
    j = ie
    m = 1
    i = ib
    iring = iperm + 1
    if (iperm.gt.7) iring = 1
    !
    ! If this segment has more than two elements  we split it
    !
    10   if (j - i - 1) 100, 90, 15
    !
    ! p is the position of an arbitrary element in the segment we choose the
    ! middle element. Under certain circumstances it may be advantageous
    ! to choose p at random.
    !
    15   p = (j + i) / 2
    ta = a(p)
    a(p) = a(i)
    go to (21, 19, 18, 17, 16, 161, 162, 163), iring
    163     th = h(p)
    h(p) = h(i)
    162     tg = g(p)
    g(p) = g(i)
    161     tf = f(p)
    f(p) = f(i)
    16      te = e(p)
    e(p) = e(i)
    17      td = d(p)
    d(p) = d(i)
    18      tc = c(p)
    c(p) = c(i)
    19      tb = b(p)
    b(p) = b(i)
    21   continue
    !
    ! Start at the beginning of the segment, search for k such that a(k)>t
    !
    q = j
    k = i
    20   k = k + 1
    if(k.gt.q)     go to 60
    if(a(k).le.ta) go to 20
    !
    ! Such an element has now been found now search for a q such that a(q)<t
    ! starting at the end of the segment.
    !
    30   continue
    if(a(q).lt.ta) go to 40
    q = q - 1
    if(q.gt.k)     go to 30
    go to 50
    !
    ! a(q) has now been found. we interchange a(q) and a(k)
    !
    40   xa = a(k)
    a(k) = a(q)
    a(q) = xa
    go to (45, 44, 43, 42, 41, 411, 412, 413), iring
    413     xh = h(k)
    h(k) = h(q)
    h(q) = xh
    412     xg = g(k)
    g(k) = g(q)
    g(q) = xg
    411     xf = f(k)
    f(k) = f(q)
    f(q) = xf
    41      xe = e(k)
    e(k) = e(q)
    e(q) = xe
    42      xd = d(k)
    d(k) = d(q)
    d(q) = xd
    43      xc = c(k)
    c(k) = c(q)
    c(q) = xc
    44      xb = b(k)
    b(k) = b(q)
    b(q) = xb
    45   continue
    !
    ! Update q and search for another pair to interchange:
    !
    q = q - 1
    go to 20
    50   q = k - 1
    60   continue
    !
    ! The upwards search has now met the downwards search:
    !
    a(i) = a(q)
    a(q) = ta
    go to (65, 64, 63, 62, 61, 611, 612, 613), iring
    613     h(i) = h(q)
    h(q) = th
    612     g(i) = g(q)
    g(q) = tg
    611     f(i) = f(q)
    f(q) = tf
    61      e(i) = e(q)
    e(q) = te
    62      d(i) = d(q)
    d(q) = td
    63      c(i) = c(q)
    c(q) = tc
    64      b(i) = b(q)
    b(q) = tb
    65   continue
    !
    ! The segment is now divided in three parts: (i,q-1),(q),(q+1,j)
    ! store the position of the largest segment in lt and ut
    !
    if (2 * q.le.i + j) go to 70
    lt(m) = i
    ut(m) = q - 1
    i = q + 1
    go to 80
    70   lt(m) = q + 1
    ut(m) = j
    j = q - 1
    !
    ! Update m and split the new smaller segment
    !
    80   m = m + 1
    go to 10
    !
    ! We arrive here if the segment has  two elements we test to see if
    ! the segment is properly ordered if not, we perform an interchange
    !
    90   continue
    if (a(i).le.a(j)) go to 100
    xa = a(i)
    a(i) = a(j)
    a(j) = xa
    go to (95, 94, 93, 92, 91, 911, 912, 913), iring
    913     xh = h(i)
    h(i) = h(j)
    h(j) = xh
    912     xg = g(i)
    g(i) = g(j)
    g(j) = xg
    911     xf = f(i)
    f(i) = f(j)
    f(j) = xf
    91    xe = e(i)
    e(i) = e(j)
    e(j) = xe
    92    xd = d(i)
    d(i) = d(j)
    d(j) = xd
    93    xc = c(i)
    c(i) = c(j)
    c(j) = xc
    94    xb = b(i)
    b(i) = b(j)
    b(j) = xb
    95 continue
    !
    ! If lt and ut contain more segments to be sorted repeat process:
    !
    100  m = m - 1
    if (m.le.0) go to 110
    i = lt(m)
    j = ut(m)
    go to 10
    110  continue
    return
end