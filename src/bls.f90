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

  subroutine eebls(n,t,x,e,freq,nb,qmi,qma,pmul,nf,p,bper,bpow,depth,qtran,in1,in2)
    !!
    !!------------------------------------------------------------------------
    !!     >>>>>>>>>>>> This routine computes BLS spectrum <<<<<<<<<<<<<<
    !!
    !!         [ see Kovacs, Zucker & Mazeh 2002, A&A, Vol. 391, 369 ]
    !!
    !!     This is the slightly modified version of the original BLS routine 
    !!     by considering Edge Effect (EE) as suggested by 
    !!     Peter R. McCullough [ pmcc@stsci.edu ].
    !!
    !!     This modification was motivated by considering the cases when 
    !!     the low state (the transit event) happened to be devided between 
    !!     the first and last bins. In these rare cases the original BLS 
    !!     yields lower detection efficiency because of the lower number of 
    !!     data points in the bin(s) covering the low state. 
    !!
    !!     For further comments/tests see  www.konkoly.hu/staff/kovacs.html
    !!------------------------------------------------------------------------
    !!
    !!     Input parameters:
    !!     ~~~~~~~~~~~~~~~~~
    !!
    !!     n    = number of data points
    !!     t    = array {t(i)}, containing the time values of the time series
    !!     x    = array {x(i)}, containing the data values of the time series
    !!     u    = temporal/work/dummy array, must be dimensioned in the 
    !!            calling program in the same way as  {t(i)}
    !!     v    = the same as  {u(i)}
    !!     freq = frequency array
    !!     nf   = number of frequency points in which the spectrum is computed
    !!     nb   = number of bins in the folded time series at any test period       
    !!     qmi  = minimum fractional transit length to be tested
    !!     qma  = maximum fractional transit length to be tested
    !!
    !!     Output parameters:
    !!     ~~~~~~~~~~~~~~~~~~
    !!
    !!     p    = array {p(i)}, containing the values of the BLS spectrum
    !!            at the i-th frequency value -- the frequency values are 
    !!            computed as  f = fmin + (i-1)*df
    !!     bper = period at the highest peak in the frequency spectrum
    !!     bpow = value of {p(i)} at the highest peak
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

    integer, intent(in) :: n, nf, nb
    real(8), intent(in) :: qmi, qma
    real(8), intent(in), dimension(n) :: t, x, e
    real(8), intent(in), dimension(nf) :: freq, pmul
    integer, intent(out) :: in1, in2
    real(8), intent(out) :: bper, bpow, depth, qtran
    real(8), intent(out), dimension(nf) :: p

    integer :: kmi, kma, i, j, k, jf, jn1, jn2
    real(8) :: rn, rn3, s, s3, ph, pow, power, f0, p0, ww, minw
    real(8), dimension(n) :: u, v, w

    real(8), dimension(:), allocatable :: y, ws
    
    minw = 0.75d0 / real(nb,8)

    !!------------------------------------------------------------------------
    !!
    rn=real(n, 8)
    kmi=int(qmi*real(nb, 8))
    if(kmi < 1) kmi=1
    kma=int(qma*real(nb, 8))+1
    bpow=0.0d0
    
    allocate(y(nb+kma), ws(nb+kma))

    !!=================================
    !!     Set temporal time series
    !!=================================
    u = t - t(1)
    v = x - sum(x)/rn
 
    !! Compute point weights
    w = e**(-2)/sum(e**(-2))

    !!******************************
    !!     Start period search     *
    !!******************************
    !$omp parallel do default(none) &
    !$omp private(i,j,k,s,ws,ww,jf,f0,p0,y,ph,pow,power,jn1,jn2,rn3,s3) &
    !$omp shared(p,u,v,w,n,nf,nb,pmul,freq,kma,kmi,qmi,minw,bpow,rn,in1,in2,qtran,depth,bper)
    do jf=1,nf
       f0 = freq(jf)
       p0 = 1.0d0/f0
       
       !!======================================================
       !!     Compute folded time series with  *p0*  period
       !!======================================================
       y  = 0.0d0
       ws = 0.0d0

       do i=1,n
          ph     = mod(u(i)*f0, 1.0d0)
          j      = 1 + int(nb*ph)
          ws(j)  = ws(j) + w(i)
          y(j)   = y(j) + w(i)*v(i)
       end do

       !!-----------------------------------------------
       !!     Extend the arrays  ibi()  and  y() beyond  
       !!     nb   by  wrapping
       !!
       do j=nb+1, nb+kma
          y(j)   = y(j-nb)
          ws(j)  = ws(j-nb)
       end do
       !!-----------------------------------------------   

       !!===============================================
       !!     Compute BLS statistics for this period
       !!===============================================
       power=0.0d0

       do i=1,nb
          s     = 0.0d0
          k     = 0
          ww    = 0.0d0 
          do j=i, i+kma
             k     = k+1
             ww    = ww+ws(j)
             s     = s+y(j)
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

       power = sqrt(power)
       power = power * pmul(jf) + (1.d0-pmul(jf))*sum(p(:jf))/real(jf,8)
       p(jf) = power

       !$omp critical
       if(power > bpow) then
          bpow  =  power
          in1   =  jn1
          in2   =  jn2
          qtran =  rn3
          depth = -s3*1.d0/(rn3*(1.d0-rn3))
          bper  =  p0
       end if
       !$omp end critical
    end do
    !$omp end parallel do

    !! Edge correction of transit end index
    if(in2 > nb) in2 = in2-nb     

    deallocate(y,ws)

  end subroutine eebls
end module bls
