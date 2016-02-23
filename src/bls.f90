!!
!! Module with routines to compute a BLS spectrum given a photometric time series.
!!
!! Adapted from the Fortran70 code available from
!!
!!  http://www.konkoly.hu/staff/kovacs/eebls.f
!!
!! Kovacs, Zucker & Mazeh 2002, A&A, Vol. 391, 369 
!!
module bls
  implicit none
  
contains
  subroutine bin(n,t,d,e,f,pc,nbin,bf,be)
    implicit none
    integer, intent(in) :: n, nbin
    real(8), intent(in), dimension(n) :: t, d, e
    real(8), intent(in) :: f, pc
    real(8), intent(out), dimension(nbin) :: bf, be

    integer :: i, id(n)
    real(8) :: w(n), dw(n)
    logical :: mask(n)

    bf  = 0.d0
    be  = 0.d0

    w   = e**(-2)/sum(e**(-2))
    id  = 1 + int(mod((t-t(1))*f + (0.5d0-pc), 1.d0) * nbin)

    dw  = d*w
    do i=1,nbin
       mask  = id == i
       bf(i) = sum(dw, mask=mask) / sum(w, mask=mask)
       be(i) = sqrt(sum(e**2, mask=mask) / real(count(mask),8))
    end do

  end subroutine bin

  subroutine eebls(np,t,x,e,freq,nb,qmi,qma,pmul,nf,p,depth,qtran,in1,in2)
    !!
    !!     Input parameters:
    !!     ~~~~~~~~~~~~~~~~~
    !!
    !!     n    = number of data points
    !!     t    = array {t(i)}, containing the time values of the time series
    !!     x    = array {x(i)}, containing the data values of the time series
    !!     e    = array {e(i)}, containing the error values of the time series
    !!     u    = temporal/work/dummy array, must be dimensioned in the 
    !!            calling program in the same way as  {t(i)}
    !!     v    = the same as  {u(i)}
    !!     freq = frequency array
    !!     nf   = number of frequency points in which the spectrum is computed
    !!     nb   = number of bins in the folded time series at any test period       
    !!     qmi  = minimum fractional transit length to be tested
    !!     qma  = maximum fractional transit length to be tested
    !!
    !!     Returns
    !!     ~~~~~~~~~~~~~~~~~~
    !!
    !!     p    = array {p(i)}, containing the values of the BLS spectrum
    !!            at the i-th frequency value -- the frequency values are 
    !!            computed as  f = fmin + (i-1)*df
    !!     depth= depth of the transit at   *bper*
    !!     qtran= fractional transit length  [ T_transit/bper ]
    !!     in1  = bin index at the start of the transit [ 0 < in1 < nb+1 ]
    !!     in2  = bin index at the end   of the transit [ 0 < in2 < nb+1 ]
    !!
    !!
    !!     Remarks:
    !!     ~~~~~~~~ 
    !!
    !!     -- *fmin* MUST be greater than  *1/total time span* 
    !!     -- *nb*   MUST be lower than  *nbmax* 
    !!     -- Dimensions of arrays {y(i)} and {ibi(i)} MUST be greater than 
    !!        or equal to  *nbmax*. 
    !!     -- The lowest number of points allowed in a single bin is equal 
    !!        to   MAX(minbin,qmi*N),  where   *qmi*  is the minimum transit 
    !!        length/trial period,   *N*  is the total number of data points,  
    !!        *minbin*  is the preset minimum number of the data points per 
    !!        bin.
    !!     
    !!========================================================================

    implicit none
    integer, parameter :: MINBIN = 10

    integer, intent(in) :: np, nf, nb
    real(8), intent(in) :: qmi, qma
    real(8), intent(in), dimension(np) :: t, x, e
    real(8), intent(in), dimension(nf) :: freq, pmul
    integer, intent(out), dimension(nf) :: in1, in2
    real(8), intent(out), dimension(nf) :: p, depth, qtran

    integer :: kmi, kma, i, j, k, jf, jn1, jn2
    real(8) :: rn3, s, s3, pow, power, ww, minw
    real(8), dimension(np) :: ntime, nflux, wght

    real(8), dimension(:), allocatable :: bflux, bwght
    
    minw = 0.75d0 / real(nb,8)
    kmi=max(1, int(qmi*real(nb, 8)))
    kma=int(qma*real(nb, 8))+1
    
    allocate(bflux(nb+kma), bwght(nb+kma))

    ntime = t - t(1)
    nflux = x - sum(x)/real(np, 8)
    wght  = e**(-2)/sum(e**(-2))

    !$omp parallel do default(none) &
    !$omp private(i,j,k,s,bwght,ww,jf,bflux,pow,power,jn1,jn2,rn3,s3) &
    !$omp shared(p,ntime,nflux,wght,np,nf,nb,pmul,freq,kma,kmi,qmi,minw,rn,in1,in2,qtran,depth)
    do jf=1,nf

       !! Fold and bin the fluxes and weights
       !! -----------------------------------
       bflux = 0.0d0
       bwght = 0.0d0
       do i=1,np
          j        = 1 + int(nb*mod(ntime(i)*freq(jf), 1.0d0))
          bwght(j) = bwght(j) + wght(i)
          bflux(j) = bflux(j) + wght(i)*nflux(i)
       end do

       !! Extend the binned arrays by wrapping
       !!-------------------------------------
       do j=nb+1, nb+kma
          bflux(j) = bflux(j-nb)
          bwght(j) = bwght(j-nb)
       end do

       !! Compute the BLS statistics
       !!---------------------------
       power=0.0d0
       do i=1,nb
          k     = 0
          s     = 0.0d0
          ww    = 0.0d0 
          do j=i, i+kma
             k     = k+1
             ww    = ww+bwght(j)
             s     = s+bflux(j)
             if ((k > kmi) .and. (ww>qmi) .and. (ww > k*minw)) then
                pow   = s*s/(ww*(1.d0-ww))

                if((pow > power) .and. (s<0.0d0)) then
                   power = pow
                   jn1   = i
                   jn2   = j
                   rn3   = ww
                   s3    = s
                end if
             end if
          end do
       end do

       !! Store the results for this period
       !! ---------------------------------
       power = sqrt(power)
       power = power * pmul(jf) + (1.d0-pmul(jf))*sum(p(:jf))/real(jf,8)
       p(jf)     = power
       in1(jf)   = jn1
       in2(jf)   = jn2
       qtran(jf) = rn3
       depth(jf) = -s3*1.d0/(rn3*(1.d0-rn3))
       if(in2(jf) > nb) in2(jf) = in2(jf)-nb     

    end do
    !$omp end parallel do

    deallocate(bflux,bwght)

  end subroutine eebls
end module bls
